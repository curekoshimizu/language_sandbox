import pathlib

import matplotlib.pyplot as plt
from PIL import Image

script_path = pathlib.Path(__file__).parent.resolve()


def test_save_choki_image() -> None:
    f = script_path.parents[0] / "datasets" / "janken_dataset" / "choki" / "IMG_0770.JPG"
    with Image.open(f) as img:
        plt.imshow(img)
        plt.savefig("choki.png")
