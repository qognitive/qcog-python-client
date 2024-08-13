
import numpy as np
import pandas as pd
import torch
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import StandardScaler


def get_wbc_data(test_fraction=0.2):
    data = sk_datasets.load_breast_cancer()

    # Keep 20% of data for testing

    assert data is not None

    n_data = data.data.shape[0]
    train_size = int(n_data * (1 - test_fraction))
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

    return df_train, df_test, df_target
