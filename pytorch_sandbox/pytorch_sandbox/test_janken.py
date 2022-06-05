import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset

from .janken import Janken, JankenDataset, Model
from .utils import Classification


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

    lr = 0.1
    batch_size = 8

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    classification = Classification(model, device, train_dataloader, test_dataloader, optimizer)
    context = classification.fit(num_epochs=100)
    figure = context.graph()
    figure.savefig("janken.png")
