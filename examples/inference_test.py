import os
import json
import pandas
import time
import numpy as np

from qcog_python_client import QcogClient

HOSTNAME = os.environ["HOSTNAME"]
API_TOKEN = os.environ["API_TOKEN"]
TRAINED_MODEL_GUID = "d9bb0c85-8d1d-4d4d-af1c-4cb203a43bc6"

hsm = QcogClient(token=API_TOKEN, hostname=HOSTNAME, verify=False).preloaded_model(TRAINED_MODEL_GUID)
print(hsm.status())

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

print(hsm.wait_for_training().inference(forecast_data, parameters))

print("done")
