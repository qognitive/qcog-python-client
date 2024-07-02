"""
In order to run the tests, you need to have a valid API_TOKEN.


**Run train test**
```bash
export API_TOKEN=<your_api_token>
pytest tests/test_examples.py -s -vv -k test_breast_cancer_train

Once the test is executed you should get the id of the trained model.

**Run inference test**
```bash
export API_TOKEN=<your_api_token>
export TRAINED_MODEL=<trained_model_id>
pytest tests/test_examples.py -s -vv -k test_breast_cancer_inference
```
"""


import os
from re import T
import numpy as np
from sklearn import datasets as sk_datasets
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from qcog_python_client import QcogClient
from qcog_python_client.schema.parameters import (
    GradOptimizationParameters,
    LOBPCGFastStateParameters,
)


API_TOKEN = os.getenv("API_TOKEN")
TRAINED_MODEL = os.getenv("TRAINED_MODEL")

data = sk_datasets.load_breast_cancer()

# Keep 20% of data for testing
TEST_FRACTION = 0.2

assert data is not None

n_data = data.data.shape[0]
train_size = int(n_data * (1 - TEST_FRACTION))
test_size = n_data - train_size
# Randomly sample dta

train_idx = np.random.choice(n_data, train_size, replace=False)
test_idx = np.random.choice(
    np.setdiff1d(np.arange(n_data), train_idx, assume_unique=True),
    test_size,
    replace=False,
)

targets = torch.nn.functional.one_hot(torch.tensor(data.target), num_classes=2).numpy()

train_data = data.data[train_idx]
train_target = targets[train_idx]
test_data = data.data[test_idx]
test_target = targets[test_idx]

# Scale data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Convert to DataFrame
df_train = pd.DataFrame(
    np.concatenate([train_data, train_target], axis=1),
    columns=data.feature_names.tolist() + data.target_names.tolist(),
)

df_test = pd.DataFrame(test_data, columns=data.feature_names)
df_target = pd.DataFrame(test_target, columns=data.target_names.tolist())


def test_breast_cancer_train():
    if API_TOKEN is None:
        raise ValueError("API_TOKEN not found in environment variables")

    qcml = QcogClient.create(
        token=API_TOKEN,
        hostname="localhost",
        port=8000,
    )

    # If a model is already trained and the guid is specified,
    # then load the model

    if TRAINED_MODEL is not None:
        qcml = qcml.preloaded_model(TRAINED_MODEL)
        # Check the status of the train
        status = qcml.status()

        if status != "completed":
            raise ValueError(f"Model is not trained yet. Status: {status}")

        else:
            print(f"Model {TRAINED_MODEL} is already trained. Skipping training.")
            return
    qcml.data(df_train)

    qcml = qcml.ensemble(
        operators=df_train.columns.tolist(),
        dim=64,
        num_axes=64,
    )

    weight_optimization = GradOptimizationParameters(
        iterations=10,
        learning_rate=1e-3,
    )

    get_states_extra = LOBPCGFastStateParameters(
        iterations=10,
    )

    qcml = qcml.train(
        batch_size=64,
        num_passes=10,
        weight_optimization=weight_optimization,
        get_states_extra=get_states_extra,
    )

    print(qcml.trained_model["guid"])


def test_breast_cancer_inference():
    qcml = QcogClient.create(
        token=API_TOKEN,
        hostname="localhost",
        port=8000,
    )

    if TRAINED_MODEL is None:
        raise ValueError("TRAINED_MODEL not found in environment variables")

    qcml.preloaded_model(TRAINED_MODEL)

    result_df = qcml.inference(
        data=df_test,
        parameters={
            "state_parameters": LOBPCGFastStateParameters(
                learning_rate_axes=1e-3,
                iterations=10,
            )
        },
    )

    num_correct = result_df.idxmax(axis=1) == df_target.idxmax(axis=1)

    print(f"Accuracy: {num_correct.sum() / len(num_correct)}")
