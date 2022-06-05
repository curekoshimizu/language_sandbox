import enum
import pathlib
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

script_path = pathlib.Path(__file__).parent.resolve()


class Janken(enum.Enum):
    GU = 1
    CHOKI = 2
    PA = 3


class JankenDataset(Dataset[tuple[torch.Tensor, Janken]]):
    def __init__(
        self,
        input_size: Optional[tuple[int, int]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
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
        if input_size is None:
            input_size = (64, 64)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self._transform = transform
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Janken]:
        assert 0 <= index < len(self)
        image_path, label = self._data[index]

        image = Image.open(image_path)
        image = self._transform(image)
        assert isinstance(image, torch.Tensor)
        return image, label


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        num_classes = 3
        self.conv1 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 32 = 64(IMAGE_SIZE) / 2
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
