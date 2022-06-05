import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import Classification


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_input = 784
        n_hidden = 128
        n_output = 10

        self.l1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(in_features=n_hidden, out_features=n_output)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


def test_mnist() -> None:
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    batch_size = 500
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    classification = Classification(
        net,
        device,
        train_loader,
        test_loader,
        optimizer,
    )
    context = classification.fit(num_epochs=100)
    figure = context.graph()
    figure.savefig("mnist.png")
