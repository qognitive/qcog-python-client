from torch.nn import Linear, Module, functional


class Net(Module):
    """Test Net class."""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(30, 100)
        self.fc2 = Linear(100, 2)

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
