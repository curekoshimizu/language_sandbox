import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn, optim


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        ret = self.sigmoid(x)
        assert isinstance(ret, torch.Tensor)
        return ret


def test_iris_prediction(save: bool = True) -> None:
    torch.manual_seed(123)

    iris = load_iris()
    x_orgin = iris.data
    y_origin = iris.target

    assert x_orgin.shape == (150, 4)
    assert y_origin.shape == (150,)

    # choose only class0 or class1 data
    indexes = (y_origin == 0) | (y_origin == 1)
    x_data = x_orgin[indexes, :2]
    y_data = y_origin[indexes]
    assert np.count_nonzero((y_data == 0) | (y_data == 1)) == 100

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=70,
        test_size=30,
    )

    net = Net()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    inputs = torch.tensor(x_train).float()
    labels = torch.tensor(y_train).float().reshape((-1, 1))
    labels1 = torch.tensor(y_train == 1).float().reshape((-1, 1))
    assert inputs.shape == (70, 2)
    assert labels.shape == (70, 1)

    inputs_test = torch.tensor(x_test).float()
    labels_test = torch.tensor(y_test).float().reshape((-1, 1))
    labels1_test = torch.tensor(y_test == 1).float().reshape((-1, 1))
    assert inputs_test.shape == (30, 2)
    assert labels_test.shape == (30, 1)

    history = []
    prev_loss = 0.0
    num_epochs = 10000
    for epoch in range(num_epochs):
        # train
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.where(outputs < 0.5, 0, 1)
        acc_train = ((predicted == labels1).sum() / len(y_train)).item()

        # test
        outputs_test = net(inputs_test)
        loss_test = criterion(outputs_test, labels_test).item()
        predicted_test = torch.where(outputs_test < 0.5, 0, 1)
        acc_test = ((predicted_test == labels1_test).sum() / len(y_test)).item()

        assert isinstance(loss, torch.Tensor) and loss.ndim == 0
        # that's why we need to call .item()
        current_loss = loss.item()
        if epoch % 100 == 0:
            print(
                "epoch : ", epoch, "loss", current_loss, "acc", acc_train, "loss_test", loss_test, "acc_test", acc_test
            )
            if abs(prev_loss - current_loss) < 1.0e-7:
                break
            prev_loss = current_loss
        history.append((epoch, current_loss, acc_train, loss_test, acc_test))

    if save:
        figure = plt.figure()
        figure1 = figure.add_subplot(2, 2, 1)

        x0 = x_train[y_train == 0]  # class 0
        figure1.scatter(x0[:, 0], x0[:, 1], marker="x")
        x1 = x_train[y_train == 1]  # class 1
        figure1.scatter(x1[:, 0], x1[:, 1], marker="*")
        figure1.set_xlabel("sepal_length")
        figure1.set_ylabel("sepal_width")

        bias = net.l1.bias.data.numpy()
        weight = net.l1.weight.data.numpy()

        xl = np.array([x_test[:, 0].min(), x_test[:, 0].max()])
        yl = -(bias + weight[0, 0] * xl) / weight[0, 1]
        figure1.plot(xl, yl, color="black")
        figure1.legend()

        figure2 = figure.add_subplot(2, 2, 2)
        history_array = np.array(history)
        figure2.plot(history_array[:, 0], history_array[:, 1], color="green")  # train loss
        figure2.plot(history_array[:, 0], history_array[:, 3], color="blue")  # test loss

        figure3 = figure.add_subplot(2, 2, 3)
        history_array = np.array(history)
        figure3.plot(history_array[:, 0], history_array[:, 2], color="green")  # train acc
        figure3.plot(history_array[:, 0], history_array[:, 4], color="blue")  # test acc

        figure.savefig("sepal.png")
