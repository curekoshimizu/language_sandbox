import numpy as np
import torch
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
