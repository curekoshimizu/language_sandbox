from typing import Optional, TypeVar

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

T = TypeVar("T")


class FitContext:
    def __init__(self) -> None:
        self._len = 0
        self._train_loss: list[float] = []
        self._train_acc: list[float] = []
        self._test_loss: list[float] = []
        self._test_acc: list[float] = []

    def append(
        self,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
    ) -> None:
        self._len += 1
        self._train_loss.append(train_loss)
        self._train_acc.append(train_acc)
        self._test_loss.append(test_loss)
        self._test_acc.append(test_acc)

    def __len__(self) -> int:
        return self._len

    def graph(self) -> plt.figure:
        figure = plt.figure()
        figure1 = figure.add_subplot(2, 1, 1)
        figure1.plot(range(1, len(self) + 1), self._train_loss, label="train")
        figure1.plot(range(1, len(self) + 1), self._test_loss, label="test")
        figure1.legend(loc="lower left")

        figure2 = figure.add_subplot(2, 1, 2)
        figure2.plot(range(1, len(self) + 1), self._train_acc, label="train")
        figure2.plot(range(1, len(self) + 1), self._test_acc, label="test")
        figure2.legend(loc="upper left")

        return figure


class Classification:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_dataloader: DataLoader[T],
        test_dataloader: DataLoader[T],
        optimizer: optim.Optimizer,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        self._model = model
        self._device = device
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._optimizer = optimizer
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self._criterion = criterion

    def _train(self) -> tuple[float, float]:
        return train(self._model, self._device, self._train_dataloader, self._optimizer, self._criterion)

    def _test(self) -> tuple[float, float]:
        return validate(self._model, self._device, self._test_dataloader, self._criterion)

    def fit(self, num_epochs: int, verbose: bool = True) -> FitContext:
        context = FitContext()
        for epoch in range(num_epochs):
            train_loss, train_acc = self._train()
            test_loss, test_acc = self._test()

            if verbose:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} test_loss: {test_loss:.5f}, test_acc: {test_acc:.5f}"
                )
            context.append(train_loss, train_acc, test_loss, test_acc)
        return context


def train(
    model: nn.Module,
    device: torch.device,
    train_dataloader: DataLoader[T],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    train_loss = 0.0
    train_acc = torch.tensor(0.0).to(device)
    n_train = 0

    model.train()
    for inputs, labels in tqdm(train_dataloader):
        assert labels.ndim == 1
        n_train += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.shape[0] == labels.shape[0]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():  # type:ignore
            _, pred = torch.max(outputs, 1)
            train_loss += loss.item()
            train_acc += (pred == labels).sum()

    train_loss /= len(train_dataloader)  # same as train_loss = train_loss * batch_size / n_train
    train_acc /= n_train

    return train_loss, train_acc.item()


def validate(
    model: nn.Module,
    device: torch.device,
    test_dataloader: DataLoader[T],
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0.0
    test_acc = torch.tensor(0.0).to(device)
    n_test = 0

    with torch.no_grad():  # type: ignore
        for inputs, labels in test_dataloader:
            assert labels.ndim == 1
            n_test += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs, 1)
            test_loss += loss.item()
            test_acc += (pred == labels).sum()

    test_loss /= len(test_dataloader)  # same as test_loss = test_loss * batch_size / n_test
    test_acc /= n_test

    return test_loss, test_acc.item()
