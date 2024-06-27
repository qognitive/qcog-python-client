import os
import pandas

from qcog_python_client import AsyncQcogClient, QcogClient


HOSTNAME = os.environ["HOSTNAME"]
API_TOKEN = os.environ["API_TOKEN"]
HOSTPORT = os.environ.get("HOSTPORT", 443)
SECURE_MODE = os.environ.get("SECURE_MODE", "true").lower() == "true"

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pandas.read_json(os.path.join(dir_path, "small0.json"))

training_parameters = {
    "batch_size": 1000,
    "num_passes": 10,
    "weight_optimization": {
        "learning_rate": 1e-3,
        "iterations": 10,
        "optimization_method": "GRAD",
        "step_size": 0.01,
        "first_moment_decay": 0.6,
        "second_moment_decay": 0.7,
        "epsilon": 1e-6
    },
    "get_states_extra": {
        "state_method": "LOBPCG_FAST",
        "iterations": 10,
        "learning_rate_axes": 0.01,
    }
}


def main():
    hsm = QcogClient.create(
        token=API_TOKEN
    ).ensemble(
        operators=["X", "Y", "Z"],
        dim=4,
        num_axes=16
    ).data(df).train(**training_parameters)

    print(hsm.trained_model)
    return hsm.trained_model["guid"]


async def async_main():
    hsm = (await AsyncQcogClient.create(
        token=API_TOKEN
    )).ensemble(
        operators=["X", "Y", "Z"],
        dim=4,
        num_axes=16
    )
    await hsm.data(df)
    await hsm.train(**training_parameters)
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
