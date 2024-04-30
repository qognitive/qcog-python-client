import os
import json
import pandas

from qcog_python_client import QcogClient, ModelClient, EnsembleInterface, PauliInterface, TrainingParameters


HOSTNAME = replace me
API_TOKEN = replace me


df = pandas.read_json("small0.json")

qcog_client = QcogClient(API_TOKEN, HOSTNAME, verify=False)

model_params = EnsembleInterface(
    {
        "operators": ["X", "Y", "Z"],
        "dim": 16,
        "num_axes": 4,
        "sigma_sq": {},
        "sigma_sq_optimization_kwargs": {},
        "seed": 42,
        "target_operators": []
    }
)

print(model_params)

training_parameters = TrainingParameters(
    {
        "batch_size": 1000,
        "num_passes": 10,
        "weight_optimization_kwargs": {
            "learning_rate": 1e-3,
            "iterations": 10,
            "optimization_method": "GRAD",
            "step_size": 0.01,
            "first_moment_decay": 0.6,
            "second_moment_decay": 0.7,
            "epsilon": 1e-6
        },
        "state_kwargs": {
            "state_method": "LOBPCG_FAST",
            "iterations": 10,
            "learning_rate_axes": 0.01,
            "fisher_axes_kwargs": {
                 "learning_rate": 1e-5
            },
            "fisher_state_kwargs": {
                 "learning_rate": 1e-5
            }
        }
    }
)

print(training_parameters)

print(df)

hsm = ModelClient(client=qcog_client).model_params("ensemble", model_params).data(df)

#hsm.preloaded_data(data guid)

print("data uploaded")

print(hsm.train(training_parameters).trained_model)

print("done")
