import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torchviz import make_dot


def test_call_backward(save_graph: bool = False) -> None:
    x_np = np.arange(-2, 2.1, 0.25)

    x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)
    assert x.dtype == torch.float32

    for i in range(5):
        y = 2 * x * x + 2
        z = y.sum()
        dot = make_dot(z, params={"x": x})
        if save_graph:
            dot.format = "png"
            dot.render("graph_image")

        z.backward()  # type: ignore
        assert torch.all(
            x.grad
            == torch.tensor(
                [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            )
        )
        x.grad.zero_()


def test_grad() -> None:
    torch.manual_seed(123)

    sample_data = np.array(
        [
            [166, 58.7],
            [176, 75.7],
            [171, 62.1],
            [173, 70.4],
            [169, 60.1],
        ]
    )

    x_origin = sample_data[:, 0]
    y_origin = sample_data[:, 1]

    x = torch.tensor((x_origin - x_origin.mean()) / x_origin.std()).float().reshape(-1, 1)
    y = torch.tensor((y_origin - y_origin.mean()) / y_origin.std()).float().reshape(-1, 1)

    l1 = nn.Linear(1, 1)
    criterion = nn.MSELoss()

    num_epochs = 500
    optimizer = SGD(l1.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        y_p = l1(x)
        loss = criterion(y_p, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print(f"epoch : {epoch}, loss : {loss}")

    for name, tensor in l1.named_parameters():
        print(name, tensor, tensor.shape)
    assert 0.1 < loss < 0.12
