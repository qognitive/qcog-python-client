import os
import json
import pandas
import time
import numpy as np

from qcog_python_client import QcogClient, ModelClient

HOSTNAME = replace me
API_TOKEN = replace me
TRAINED_MODEL_GUID = replace me


qcog_client = QcogClient(API_TOKEN, HOSTNAME, verify=False)

hsm = ModelClient.from_model_guid(TRAINED_MODEL_GUID, client=qcog_client, with_data=True)
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
