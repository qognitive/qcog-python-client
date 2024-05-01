import os
import json
import pandas
import time
import numpy as np

from qcog_python_client import QcogClient

HOSTNAME = os.environ["HOSTNAME"]
API_TOKEN = os.environ["API_TOKEN"]
TRAINED_MODEL_GUID = "b9cb6828-9aca-44c8-991e-19581487e1fc"  # "20c0353d-05f1-43f4-864d-4ab9f8e659c1" # replace me

hsm = QcogClient(token=API_TOKEN, hostname=HOSTNAME, verify=False).preloaded_model(TRAINED_MODEL_GUID)
print(hsm.status())

while True:
    if hsm.status()["status"] == "completed":
        break
    print(hsm.status())
    time.sleep(5)

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

print(hsm.inference(forecast_data, parameters))

print("done")
