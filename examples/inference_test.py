"""Example of inference with a preloaded model."""

import os

import numpy as np
import pandas

from qcog_python_client import AsyncQcogClient, QcogClient
from qcog_python_client.schema import LOBPCGFastStateParameters

API_TOKEN = os.environ["API_TOKEN"]
TRAINED_MODEL_GUID = os.environ.get(
    "TRAINED_MODEL", "c491b576-f74c-3790-a8e1-c13c55cb528d"
)

print("################################")
print("# FORECAST                     #")
print("################################")

n_forecastdata = 10
xis_forecast = np.random.randn(n_forecastdata, 2)
xs_forecast = np.ones(n_forecastdata)
xs_forecast[xis_forecast[:, 0] < 0] = -1
ys_forecast = np.ones(n_forecastdata)
ys_forecast[xis_forecast[:, 1] < 0] = -1
forecast_data = pandas.DataFrame(
    data=(np.vstack([xs_forecast, ys_forecast])).T,
    index=range(n_forecastdata),
    columns=["X", "Y"],
)


def main():
    """Run training."""
    hsm = QcogClient.create(
        token=API_TOKEN,
        version="0.0.75",
        hostname="localhost",
        port=80,
    ).preloaded_model(TRAINED_MODEL_GUID)
    print(hsm.status())
    print(
        hsm.wait_for_training().inference(
            forecast_data,
            {
                "state_parameters": LOBPCGFastStateParameters(
                    iterations=5,
                    learning_rate_axes=0,
                )
            },
        )
    )


async def async_main():
    """Run training async."""
    hsm = await (await AsyncQcogClient.create(token=API_TOKEN)).preloaded_model(
        TRAINED_MODEL_GUID
    )
    print(await hsm.status())
    await hsm.wait_for_training()
    print(
        await hsm.inference(
            forecast_data,
            {
                "state_parameters": LOBPCGFastStateParameters(
                    iterations=5,
                    learning_rate_axes=0,
                )
            },
        )
    )


if __name__ == "__main__":
    print("################################")
    print("# SYNC                         #")
    print("################################")
    main()
    print("################################")
    print("# ASYNC                        #")
    print("################################")
    import asyncio

    asyncio.run(async_main())
    print("done")
