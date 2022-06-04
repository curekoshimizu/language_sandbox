import enum
import pathlib

import matplotlib.pyplot as plt
# import numpy as np
from PIL import Image
from torch.utils.data import Dataset

script_path = pathlib.Path(__file__).parent.resolve()


def test_save_choki_image() -> None:
    f = script_path.parents[0] / "datasets" / "janken_dataset" / "choki" / "IMG_0770.JPG"
    with Image.open(f) as img:
        plt.imshow(img)
        plt.savefig("choki.png")


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
        image_path, label = self._data[index]

        image = Image.open(image_path)
        # image = image.resize(self.input_size)  # リサイズ
        # image = np.array(image).astype(np.float32).transpose(2, 1, 0)  # Dataloader で使うために転置する

        return image, label


def test_dataloader() -> None:
    dataset = JankenDataset()
    assert len(dataset) == 166
