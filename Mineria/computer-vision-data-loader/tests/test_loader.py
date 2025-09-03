import csv
import io
import pathlib
import uuid

import pytest
import numpy as np
from PIL import Image

from src import loader


def test_loads_all_data(tmp_path: pathlib.Path):
    image_1 = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
    image_2 = np.random.randint(0, 255, (10, 20, 3), dtype=np.uint8)
    source_1, sprite_1, image_bytes_1 = _build_source(tmp_path, image_1, "Bulbasaur")
    source_2, sprite_2, image_bytes_2 = _build_source(tmp_path, image_2, "Charmander")
    iterator = loader.load(
        [source_1, source_2],
        downloader=FakeDownloader({sprite_1: image_bytes_1, sprite_2: image_bytes_2}),
    )

    row = next(iterator)
    np.testing.assert_array_equal(row.image, image_1)
    assert row.name == "Bulbasaur"

    row = next(iterator)
    np.testing.assert_array_equal(row.image, image_2)
    assert row.name == "Charmander"

    with pytest.raises(StopIteration):
        next(iterator)


def _build_source(
    dir: pathlib.Path, image: np.ndarray, name: str
) -> tuple[pathlib.Path, str, bytes]:
    source = dir / f"{uuid.uuid4()}.csv"
    sprite = dir / f"{uuid.uuid4()}.png"
    im = Image.fromarray(image)
    image_bytes = io.BytesIO()
    im.save(image_bytes, format="png")
    with open(source, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Pokemon", "Sprite"])
        writer.writeheader()
        writer.writerow({"Pokemon": name, "Sprite": sprite})
    return source, str(sprite), image_bytes.getvalue()


class FakeDownloader:
    def __init__(self, data: dict[str, bytes]):
        self.data = data

    def __call__(self, key: str) -> bytes:
        return self.data[key]
