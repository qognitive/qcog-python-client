"""Example of training a model."""

import os

import pandas

from qcog_python_client import AsyncQcogClient, QcogClient
from qcog_python_client.schema import GradOptimizationParameters, GradStateParameters

API_TOKEN = os.environ["API_TOKEN"]

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pandas.read_json(os.path.join(dir_path, "small0.json"))


states_extra = GradStateParameters(
    iterations=10,
    learning_rate=0.01,
)


def main():
    """Run training."""
    hsm = (
        QcogClient.create(
            token=API_TOKEN,
            version="0.0.71",
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
    return hsm.trained_model["guid"]


async def async_main():
    """Run training async."""
    hsm = (await AsyncQcogClient.create(token=API_TOKEN)).ensemble(
        operators=["X", "Y", "Z"], dim=4, num_axes=16
    )
    await hsm.data(df)
    await hsm.train(
        batch_size=1000,
        num_passes=10,
        weight_optimization=GradOptimizationParameters(
            iterations=10,
            learning_rate=1e-3,
        ),
        get_states_extra=states_extra,
    )
    print(hsm.trained_model)
    return hsm.trained_model["guid"]


if __name__ == "__main__":
    print("################################")
    print("# SYNC                         #")
    print("################################")
    guid = main()
    print("################################")
    print("# ASYNC                        #")
    print("################################")
    import asyncio

    asyncio.run(async_main())
    print("done")

    print(f"\nexport TRAINED_MODEL={guid}")
