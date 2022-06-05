from typing import TypeVar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

T = TypeVar("T")


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
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():  # type:ignore
            _, pred = torch.max(outputs, 1)
            train_loss += loss.item()
            train_acc += (pred == labels).sum()

    train_loss /= n_train
    train_acc /= n_train

    return train_loss, train_acc.item()


def test(
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

    test_loss /= n_test
    test_acc /= n_test

    return test_loss, test_acc.item()
