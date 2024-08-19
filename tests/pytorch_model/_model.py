import torch


class Model(torch.nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(dim, 16)
        self.l2 = torch.nn.Linear(16, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
