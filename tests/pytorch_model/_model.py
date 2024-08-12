import torch


class Net(torch.nn.Module):
    """Test Net class."""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(30, 100)
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
