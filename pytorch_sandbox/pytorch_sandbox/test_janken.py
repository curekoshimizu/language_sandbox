import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

from .janken import Janken, JankenDataset


def test_dataloader(seed: int = 100) -> None:
    image_dataset = JankenDataset()
    assert len(image_dataset) == 166

    num_train = int(len(image_dataset) * 0.7)
    num_valid = len(image_dataset) - num_train

    train_dataset, valid_dataset = random_split(
        image_dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(seed)
    )
    assert len(train_dataset) == num_train
    assert len(valid_dataset) == num_valid


def test_save_image() -> None:
    dataset = JankenDataset()
    img, gu = dataset[0]
    assert img.shape == (3, 64, 64)
    assert gu == Janken.GU
    # How to display torch.Tensor
    # https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
    assert img.permute(1, 2, 0).shape == (64, 64, 3)
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("gu.png")
