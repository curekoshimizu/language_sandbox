import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset

from .janken import Janken, JankenDataset, Model
from .utils import train


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

    LEARNING_RATE = 1.0
    EPOCHS = 20
    BATCH_SIZE = 4

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # TODO: REMOVE
    model = Model().to(device)
    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    def test(
        model: Model, device: torch.device, test_dataloader: DataLoader[tuple[torch.Tensor, int]]
    ) -> tuple[float, float]:
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        correct = 0.0
        total = 0.0

        with torch.no_grad():  # type:ignore
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.nll_loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_loss = val_loss / len(test_dataloader)
        val_acc = correct / total

        return val_loss, val_acc

    train_loss_list = []

    val_loss_list = []
    val_acc_list = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, device, train_dataloader, optimizer)
        val_loss, val_acc = test(model, device, test_dataloader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print("epoch: {:d}".format(epoch))
        print("val_loss: {:.4f}, val_acc: {:.4f}".format(100.0 * val_loss, val_acc))
