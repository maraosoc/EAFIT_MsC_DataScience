from dataclasses import dataclass
from typing import Iterator, List
import pathlib
import matplotlib.pyplot as plt
import time
import os
import csv
from PIL import Image

@dataclass
class Row:
    name: str
    image: object
    path: pathlib.Path
    type1: str

# Leer el csv 
def rows_from_csv(csv_files: List[str], output_dir: str) -> Iterator[Row]:
    out = pathlib.Path(output_dir)
    for filepath in csv_files:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name  = r["Pokemon"].lower()
                type1 = r["Type1"].lower()
                p = out / type1 / f"{name}.png"
                if not p.exists():
                    continue  # Por si falla la descarga
                img = Image.open(p).convert("RGBA")
                yield Row(name=name, image=img, path=p, type1=type1)

def produce(csvs, output_dir):
    iterator = rows_from_csv(csvs, output_dir)
    t0 = time.perf_counter()
    row = next(iterator)
    print(f"La operación tómo {time.perf_counter() - t0:.5f}s")
    plt.imshow(row.image)
    plt.title(row.name)
    return row

if __name__ == "__main__":
    import glob
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    output_dir = os.path.join(base_dir, 'output/iterator/pokemons')
    produce(csv_files, output_dir)
