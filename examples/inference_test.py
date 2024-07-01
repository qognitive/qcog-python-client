import os
import pandas
import numpy as np

from qcog_python_client import QcogClient, AsyncQcogClient

API_TOKEN = os.environ["API_TOKEN"]
TRAINED_MODEL_GUID = os.environ.get("TRAINED_MODEL", "9d1432c3-6f43-4c90-aa32-1ac2d54e9a3c")

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
    columns=["X", "Y"]
)
parameters = {
    "state_method": "LOBPCG_FAST",
    "iterations": 5,
    "learning_rate_axes": 0,
    "learning_rate_psi": 0.1
}


def main():
    hsm = QcogClient.create(
        token=API_TOKEN,
        hostname="127.0.0.1",
        port=8000
    ).preloaded_model(TRAINED_MODEL_GUID)
    print(hsm.status())
    print(hsm.wait_for_training().inference(forecast_data, parameters))


async def async_main():
    hsm = (
        await (
            await AsyncQcogClient.create(
                token=API_TOKEN
            )
        ).preloaded_model(TRAINED_MODEL_GUID)

    )
    print(await hsm.status())
    await hsm.wait_for_training()
    print(await hsm.inference(forecast_data, parameters))


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
