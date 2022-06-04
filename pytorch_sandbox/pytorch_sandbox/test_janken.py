import enum
import pathlib

import matplotlib.pyplot as plt
import torch
# import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split

script_path = pathlib.Path(__file__).parent.resolve()


class Janken(enum.Enum):
    GU = 1
    CHOKI = 2
    PA = 3


class JankenDataset(Dataset[tuple[Image.Image, Janken]]):
    def __init__(self) -> None:
        super().__init__()
        root_dir = script_path.parents[0] / "datasets" / "janken_dataset"
        gu = root_dir / "gu"
        choki = root_dir / "choki"
        pa = root_dir / "pa"

        data = []
        for d in gu.glob("*.JPG"):
            data.append((d, Janken.GU))
        for d in choki.glob("*.JPG"):
            data.append((d, Janken.CHOKI))
        for d in pa.glob("*.JPG"):
            data.append((d, Janken.PA))
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[Image.Image, Janken]:
        assert 0 <= index < len(self)
        image_path, label = self._data[index]

        image = Image.open(image_path)
        # image = image.resize(self.input_size)  # リサイズ
        # image = np.array(image).astype(np.float32).transpose(2, 1, 0)  # Dataloader で使うために転置する

        return image, label


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
    assert gu == Janken.GU
    plt.imshow(img)
    plt.savefig("gu.png")
