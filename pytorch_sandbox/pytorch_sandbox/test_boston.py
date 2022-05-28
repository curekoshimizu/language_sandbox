import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_boston
from torch import nn, optim


class Net(nn.Module):
    def __init__(self, n_input: int, n_output: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.l1(x)
        assert isinstance(x1, torch.Tensor)
        return x1


def test_boston_prediction(save: bool = False) -> None:
    torch.manual_seed(123)
    boston = load_boston()

    x_orgin = boston.data
    y = boston.target
    feature_names = boston.feature_names

    assert x_orgin.shape == (506, 13)
    assert y.shape == (506,)
    assert feature_names.shape == (13,)

    x = x_orgin[:, feature_names == "RM"]
    assert x.shape == (506, 1)

    net = Net(n_input=1, n_output=1)

    def show_params() -> None:
        print("---------------------------")
        for param in net.named_parameters():
            print(param)
        print("---------------------------")

    # display current setting
    show_params()

    criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    inputs = torch.tensor(x).float()
    labels = torch.tensor(y).float().reshape((-1, 1))
    assert inputs.shape == (506, 1)
    assert labels.shape == (506, 1)

    prev_loss = 0.0
    num_epochs = 50000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            assert isinstance(loss, torch.Tensor) and loss.ndim == 0
            # that's why we need to call .item()
            current_loss = loss.item()
            print("epoch : ", epoch, "loss", current_loss)
            if abs(prev_loss - current_loss) < 1.0e-7:
                break
            prev_loss = current_loss

    # display current setting
    show_params()

    xse = torch.tensor(np.array((x.min(), x.max())).reshape(-1, 1)).float()
    with torch.no_grad():  # type:ignore
        yse = net(xse)

    if save:
        plt.scatter(x, y, s=10, c="blue")  # s: size  # c: color
        plt.plot(xse.data, yse.data, c="black")
        plt.savefig("room_price.png")

    assert 43 < loss.item() < 44
