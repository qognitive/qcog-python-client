import os
import json
import pandas

from qcog_python_client import QcogClient, TrainingParameters


HOSTNAME = os.environ["HOSTNAME"]
API_TOKEN = os.environ["API_TOKEN"]


df = pandas.read_json("small0.json")

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


qcog_client = QcogClient(API_TOKEN, HOSTNAME, verify=False)
hsm = QcogClient(API_TOKEN, HOSTNAME, verify=False, verbose=True).EnsembleHSM(operators=["X", "Y", "Z"]).data(df).train(**training_parameters)

print(hsm.trained_model)

# model_params = EnsembleInterface(
#     {
#         "operators": ["X", "Y", "Z"],
#         "dim": 16,
#         "num_axes": 4,
#         "sigma_sq": {},
#         "sigma_sq_optimization_kwargs": {},
#         "seed": 42,
#         "target_operators": []
#     }
# )
