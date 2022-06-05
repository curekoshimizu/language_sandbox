from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

T = TypeVar("T")


def train(
    model: nn.Module,
    device: torch.device,
    train_dataloader: DataLoader[T],
    optimizer: optim.Optimizer,
) -> float:
    train_loss = 0.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()

        loss.backward()  # type:ignore
        optimizer.step()

    train_loss /= len(train_dataloader)

    return train_loss
