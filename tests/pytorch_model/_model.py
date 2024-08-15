import torchvision.models as models
from torch import nn
from torch.optim import SGD


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Preload resnet. Supress logs while importing
        self.model = models.resnet50(weights="ResNet50_Weights.DEFAULT", progress=False)
        self.loss = nn.CrossEntropyLoss()

        # Apply our optimizer
        self.optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x, target=None):
        x = self.model(x)

        if self.training:
            loss = self.loss(x, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return x, loss
        else:
            return x
