import csv
import functools
import os
import shutil
import time
import typing as t
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Crea directorios y se asegura de que existan
def ensure_dir(dirpath: str):
    """Ensure that a directory exists, if not, create it."""
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)

def maybe_remove_dir(dirpath: str):
    """Remove a directory if it exists."""
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

def maybe_create_dir(dirpath: str):
    """Create a directory if if does not exists already."""
    if not os.path.isdir(dirpath):
        try:
            os.mkdir(dirpath)
        except FileExistsError:
            pass

def timeit(f: t.Callable):
    """Measure the execution time of a function."""
    @functools.wraps(f)
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - ts
        print(f"Elapsed is {elapsed_time:2.4f}")
        return result
    return timed


def read_csv_rows_as_dict(filepath: str):
    """Iterate over the rows of a single csv."""
    with open(filepath, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def read_all_csv_rows_as_dict(inputs: t.List[str]):
    """Iterate over all rows from all csvs."""
    for fpath in inputs:
        for row in read_csv_rows_as_dict(fpath):
            yield row

# Normaliza los nombres y tipos en minusculas
def read_pokemons(inputs: t.List[str]):
    """Read all csv inputs and make lowercase."""
    for row in read_all_csv_rows_as_dict(inputs):
        row["Type1"] = row["Type1"].lower()
        row["Pokemon"] = row["Pokemon"].lower()
        yield row

# Escribe el contenido de la imagen
def write_binary(filepath: str, content: bytes):
    """Write binary contents to a file."""
    with open(filepath, mode="wb") as f:
        f.write(content)

# Descarga usando la sesión del hilo
def maybe_download_sprite(session, sprite_url: str):
    """Return the content of a sprite if the get request is successfull.

    Note that this function is noy asynchronous, so it may be inneficient to called it
    withing an async function.
    """
    content = None
    with session.get(sprite_url, timeout=10) as response:
        if response.status_code == 200:
            content = response.content
    return content

def analyze_load_times(csv_iterative: str, csv_pipeline: str):
    """
    Analiza los tiempos de carga de imágenes desde CSV y genera estadísticas y gráficos.

    :param csv_iterative: ruta al CSV con los tiempos de carga iterativos
    :param csv_pipeline: ruta al CSV con los tiempos de carga con productor-consumidor
    """

    # ---------------- LEER CSV ----------------
    df_iter = pd.read_csv(csv_iterative)
    df_pipe = pd.read_csv(csv_pipeline)

    # Se asume que cada CSV tiene una columna "duration" con los tiempos de carga
    times_iter = df_iter["duration"]
    times_pipe = df_pipe["duration"]

    # ---------------- ESTADÍSTICAS ----------------
    stats = pd.DataFrame({
        "Iterativo": [
            times_iter.mean(),
            times_iter.median(),
            times_iter.std(),
            times_iter.min(),
            times_iter.max(),
            times_iter.sum()
        ],
        "Productor-Consumidor": [
            times_pipe.mean(),
            times_pipe.median(),
            times_pipe.std(),
            times_pipe.min(),
            times_pipe.max(),
            times_pipe.sum()
        ]
    }, index=["Media", "Mediana", "Desv. Estándar", "Mínimo", "Máximo", "Suma total"])

    print("Estadísticas de tiempos de carga (segundos):")
    print(stats)

    # ---------------- GRÁFICOS ----------------
    # Boxplot comparativo
    plt.figure(figsize=(8, 5))
    plt.boxplot([times_iter, times_pipe], labels=["Iterativo", "Productor-Consumidor"])
    plt.ylabel("Tiempo de carga (s)")
    plt.title("Comparación de tiempos de carga de imágenes")
    plt.grid(True)
    plt.show()

    # Histograma comparativo
    plt.figure(figsize=(8, 5))
    plt.hist(times_iter, bins=20, alpha=0.6, label="Iterativo")
    plt.hist(times_pipe, bins=20, alpha=0.6, label="Productor-Consumidor")
    plt.xlabel("Tiempo de carga (s)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de tiempos de carga de imágenes")
    plt.legend()
    plt.grid(True)
    plt.show()
    return stats