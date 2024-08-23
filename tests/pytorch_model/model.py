"""Test PyTorch model."""

import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from _model import Model
from sklearn.calibration import LabelEncoder
from torch.autograd import Variable


def train(
    data: pd.DataFrame,
    *,
    epochs: int,
    batch_size: int,
) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cols = data.columns

    # Show the data
    x_data = data[cols[2:-1]]
    # Drop last column
    x_data = x_data.drop(x_data.columns[-1], axis=1)
    y_data = data[cols[-2]]  # Labels
    le = LabelEncoder()
    y_data = np.array(le.fit_transform(y_data))

    print("-- Y data Shape:  ", y_data.shape)
    print("-- X data Shape:  ", x_data.values.shape)

    model = Model(dim=x_data.values.shape[1])
    model.to(device)

    x_data = Variable(torch.from_numpy(x_data.values))
    y_data = Variable(torch.from_numpy(y_data))

    criterion = torch.nn.BCELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_list = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        y_pred = model(x_data.float())
        loss = criterion(y_pred, y_data.view(-1, 1).float())
        # print('Epoch', epoch, 'Loss:',e loss.item(), '- Pred:', y_pred.data[0])
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {"model": model, "metrics": {"loss": loss_list}}
