import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset

from .janken import Janken, JankenDataset, Model
from .utils import test, train


def test_dataloader(seed: int = 100) -> None:
    image_dataset = JankenDataset()
    assert len(image_dataset) == 166

    num_train = int(len(image_dataset) * 0.7)
    num_valid = len(image_dataset) - num_train

    train_dataset, test_dataset = random_split(
        image_dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(seed)
    )
    assert len(train_dataset) == num_train
    assert len(test_dataset) == num_valid


def test_save_image() -> None:
    dataset = JankenDataset()
    img, gu = dataset[0]
    assert img.shape == (3, 64, 64)
    assert gu == Janken.GU.value
    # How to display torch.Tensor
    # https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
    assert img.permute(1, 2, 0).shape == (64, 64, 3)
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("gu.png")


def test_model(seed: int = 100) -> None:
    image_dataset = JankenDataset()
    assert len(image_dataset) == 166

    num_train = int(len(image_dataset) * 0.7)
    num_valid = len(image_dataset) - num_train

    train_dataset, test_dataset = random_split(
        image_dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(seed)
    )
    assert len(train_dataset) == num_train
    assert len(test_dataset) == num_valid
    assert isinstance(train_dataset, Subset)

    LEARNING_RATE = 0.1
    num_epochs = 100
    BATCH_SIZE = 8

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # TODO: REMOVE
    model = Model().to(device)
    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_dataloader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_dataloader, criterion)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} test_loss: {test_loss:.5f}, test_acc: {test_acc:.5f}"
        )
        history.append(([epoch + 1, train_loss, train_acc, test_loss, test_acc]))
