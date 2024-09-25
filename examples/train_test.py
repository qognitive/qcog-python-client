"""Example of training a model."""

import os
import time

import numpy as np
import pandas
from pandas import DataFrame

from qcog_python_client import AsyncQcogClient, QcogClient
from qcog_python_client.schema import GradOptimizationParameters, GradStateParameters

API_TOKEN = os.environ["API_TOKEN"]

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pandas.read_json(os.path.join(dir_path, "small0.json"))


states_extra = GradStateParameters(
    iterations=10,
    learning_rate=0.01,
)

HOST = os.getenv("QCOG_HOST", "dev.qognitive.io")
PORT = os.getenv("QCOG_PORT", 443)
DATASET_ID = os.getenv("DATASET_ID", None)
MODEL_ID = os.getenv("MODEL_ID", None)


def _get_test_df(size_mb: int) -> DataFrame:
    # Estimate the number of rows needed to reach the desired size
    # This is an approximation and may need adjustment
    row_size_bytes = 100  # Estimated size per row in bytes
    num_rows = (size_mb * 1024 * 1024) // row_size_bytes

    # Create the DataFrame
    df = DataFrame(
        {
            "id": range(num_rows),
            "float_col": np.random.rand(num_rows),
            "int_col": np.random.randint(0, 100, num_rows),
            "string_col": np.random.choice(["A", "B", "C", "D"], num_rows),
            "bool_col": np.random.choice([True, False], num_rows),
        }
    )

    # Check the actual size and adjust if necessary
    actual_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Actual DataFrame size: {actual_size_mb:.2f} MB")

    return df


def main():
    """Run training."""
    hsm = (
        QcogClient.create(
            token=API_TOKEN,
        )
        .ensemble(
            operators=["X", "Y", "Z"],
            dim=4,
            num_axes=16,
        )
        .data(df)
        .train(
            batch_size=1000,
            num_passes=10,
            weight_optimization=GradOptimizationParameters(
                iterations=10,
                learning_rate=1e-3,
            ),
            get_states_extra=states_extra,
        )
    )

    print(hsm.trained_model)
    hsm.wait_for_training(poll_time=5)
    return hsm.trained_model["guid"]


async def async_main():
    """Run training async."""
    hsm = (
        await AsyncQcogClient.create(
            token=API_TOKEN,
        )
    ).ensemble(operators=["X", "Y", "Z"], dim=4, num_axes=16)
    await hsm.data(df)
    await hsm.train(
        batch_size=10,
        num_passes=25,
        weight_optimization=GradOptimizationParameters(
            iterations=10,
            learning_rate=1e-3,
        ),
        get_states_extra=states_extra,
    )
    await hsm.wait_for_training(poll_time=5)
    loss = await hsm.loss()
    print("LOSS: ", loss)
    print(hsm.trained_model)
    return hsm.trained_model["guid"]


async def big_data_test() -> None:
    """Upload data as a stream."""
    client = await AsyncQcogClient.create(
        token=API_TOKEN,
        hostname=HOST,
        port=PORT,
    )

    if DATASET_ID is None:
        dataset_id = "big_data_36"

        big_df = _get_test_df(10000)
        size = big_df.memory_usage(deep=True).sum() / 1024**2
        print("Testing Size of big_df MB: ", size)

        print("Testing upload_data")

        start = time.time()
        await client.upload_data(big_df, dataset_id)
        end = time.time()
        print(f"`upload_data` Time taken to upload {size} MB of data: ", end - start)
    else:
        print("Using existing dataset")
        await client.preloaded_data(DATASET_ID)

    print("Test Model Training")
    client.pauli(
        operators=["X", "Y", "Z"],
    )
    await client.train(
        batch_size=1,
        num_passes=1,
        weight_optimization=GradOptimizationParameters(
            iterations=10,
            learning_rate=1e-3,
        ),
        get_states_extra=states_extra,
    )

    await client.status()
    print(client.trained_model)


async def check_status() -> None:
    """Check status."""
    client = await AsyncQcogClient.create(
        token=API_TOKEN,
        hostname=HOST,
        port=PORT,
    )
    if MODEL_ID is None:
        raise ValueError("MODEL_GUID is not set")

    await client.preloaded_model(MODEL_ID)
    status = await client.progress()
    print(status)


if __name__ == "__main__":
    import asyncio

    # print("################################")
    # print("# SYNC                         #")
    # print("################################")
    # guid = main()
    # print("################################")
    # print("# ASYNC                        #")
    # print("################################")
    # asyncio.run(async_main())
    # print("done")
    # print(f"\nexport TRAINED_MODEL={guid}")
    print("################################")
    print("# UPLOAD STREAM                #")
    print("################################")
    # asyncio.run(big_data_test())
    asyncio.run(check_status())
