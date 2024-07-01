import os
import numpy as np
from sklearn import datasets as sk_datasets
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from qcog_python_client import QcogClient
from qcog_python_client.schema.parameters import GradOptimizationParameters, LOBPCGFastStateParameters


def test_breast_cancer():
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

    targets = torch.nn.functional.one_hot(
        torch.tensor(data.target), num_classes=2
    ).numpy()

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

    API_TOKEN = os.getenv("API_TOKEN")

    if API_TOKEN is None:
        raise ValueError("API_TOKEN not found in environment variables")

    qcml = QcogClient.create(
        token=API_TOKEN,
        hostname="localhost",
        port=8000,
    )

    qcml.data(df_train)

    qcml = qcml.ensemble(
        operators=df_train.columns.tolist(),
        dim=64,
        num_axes=64,
    )


    # weight_optimization={
    #     "learning_rate": 1e-3,
    #     "iterations": 5,
    #     "optimization_method": "GRAD"
    # },

    weight_optimization=GradOptimizationParameters(
        iterations=10,
        learning_rate=1e-3,
    )

    get_states_extra=LOBPCGFastStateParameters(
        iterations=10,
    )

    qcml = qcml.train(
        batch_size=64,
        num_passes=10,
        weight_optimization=weight_optimization,
        get_states_extra=get_states_extra
    )

    qcml.wait_for_training()
    print(qcml.trained_model["guid"])


if __name__ == "__main__":
    test_breast_cancer()