import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 100
    history = []

    for epoch in range(num_epochs):
        # train
        train_acc = torch.tensor(0.0).to(device)
        train_loss = 0.0
        n_train = 0
        for inputs, labels in tqdm(train_loader):
            assert inputs.shape == (500, 784)
            assert labels.shape == (500,)

            n_train += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():  # type:ignore
                _, pred = torch.max(outputs, 1)
                train_loss += loss.item()
                train_acc += (pred == labels).sum()

        with torch.no_grad():  # type:ignore
            # test
            test_acc = torch.tensor(0.0).to(device)
            test_loss = 0.0
            n_test = 0
            for inputs, labels in test_loader:
                n_test += len(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, pred = torch.max(outputs, 1)
                test_loss += loss.item()
                test_acc += (pred == labels).sum()

            train_acc = train_acc / n_train
            test_acc = test_acc / n_test
            train_loss = train_loss * batch_size / n_train
            test_loss = test_loss * batch_size / n_test
            print(
                f"Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} test_loss: {test_loss:.5f}, test_acc: {test_acc:.5f}"
            )
            history.append(([epoch + 1, train_loss, train_acc.item(), test_loss, test_acc.item()]))

        history_array = np.array(history)

        figure = plt.figure()
        figure1 = figure.add_subplot(2, 1, 1)
        figure1.plot(history_array[:, 0], history_array[:, 1], label="train")
        figure1.plot(history_array[:, 0], history_array[:, 3], label="test")

        figure2 = figure.add_subplot(2, 1, 2)
        figure2.plot(history_array[:, 0], history_array[:, 2], label="train")
        figure2.plot(history_array[:, 0], history_array[:, 4], label="test")

        figure.savefig("mnist.png")
