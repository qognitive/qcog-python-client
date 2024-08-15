"""Test PyTorch model."""

import json

import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from pytorch_model._model import Net


def train(
    trainset: torchvision.datasets.CIFAR10,
    *,
    epochs: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    """Test Training function."""
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    net = Net()

    net.to(device)
    print(f"Moved Resnet model to {device}")

    print("Starting the model training!")
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for _, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            _, loss = net(inputs, labels)

            # print statistics
            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.3f}"
        )

    print("Finished model training")

    print("Evaluating the model performance")
    net.eval()

    # Test accuracy
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the retrained Resnet model on all 10000 test images: {100 * correct // total} %"  # noqa: E501
    )

    return {"model": net, "metrics": json.dumps({"accuracy": 100 * correct // total})}


class TestMyClass:
    pass
