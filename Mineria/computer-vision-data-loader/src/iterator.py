from dataclasses import dataclass
from typing import Iterator, List
import pathlib
import matplotlib.pyplot as plt
import time
import os
import csv
from PIL import Image
import threading
import queue
import logging
import glob
from utils import ensure_dir
# ---------------- CONFIGURACIÓN LOGGING ----------------
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S")


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
                    logging.warning(f"{p} does not exist.")
                    continue
                img = Image.open(p).convert("RGBA")
                yield Row(name=name, image=img, path=p, type1=type1)

def test_iterator(csv_files, output_dir):
    iterator = rows_from_csv(csv_files, output_dir)
    try:
        t0 = time.perf_counter()
        row = next(iterator)
        duration = time.perf_counter() - t0
        logging.info(f"Iterador básico: imagen ({row.name}) cargada en {duration:.5f}s")
        plt.title(row.name)
        plt.imshow(row.image)
        plt.show()
        return row
    except StopIteration:
        logging.error("No hay imágenes válidas en la ruta.")
        return None

# ------------------ PRODUCTOR Y CONSUMIDOR ------------------
def test_pipeline(csv_files, output_dir, maxsize=20):
    # Cola compartida entre productor y consumidor
    pipeline = queue.Queue(maxsize=maxsize)
    event = threading.Event()
# --------------- PRODUCTOR DE LECTURA ---------------
    def producer(csv_files: List[str], output_dir: str):
        try:
            for row in rows_from_csv(csv_files, output_dir):
                pipeline.put(row)  # bloquea si la cola está llena
                logging.info(f"Productor: {row.name} añadido a la fila")
        except Exception as e:
            logging.error(f"Error en el productor: {e}")
        finally:
            event.set()  # señal de que no habrá más filas
            logging.info("Productor: terminó, evento establecido")

# --------------- CONSUMIDOR DE LECTURA ---------------
    def consumer():
        t0 = time.perf_counter()
        processed_count = 0

        while not event.is_set() or not pipeline.empty():
            try:
                row = pipeline.get(timeout=0.1)
                processed_count += 1
                duration = time.perf_counter() - t0
                logging.info(f"Productor-Consumidor: imagen ({row.name}) cargada en {duration:.5f}s")

                # Solo mostrar la primera imagen procesada
                if processed_count == 1:
                    plt.title(f"{row.name} tipo ({row.type1})")
                    plt.imshow(row.image)
                    plt.show()
                pipeline.task_done()
            except queue.Empty:
                if event.is_set():   # si el evento indica que no hay más datos
                    logging.info("Consumidor: evento establecido y cola vacía, terminando.")
                    break
                continue

        logging.info(f"Consumidor: procesó {processed_count} imágenes.")

    # Crear y ejecutar los hilos
    producer_thread = threading.Thread(target=producer, args=(csv_files, output_dir), name="Producer")
    consumer_thread = threading.Thread(target=consumer, name="Consumer", daemon=True)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()
    logging.info("Procesamiento completo.")

# ------------------ MAIN ------------------

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    output_dir = os.path.join(base_dir, 'output','pokemons')
    ensure_dir(output_dir)

    logging.info(f"Archivos CSV encontrados: {len(csv_files)}")
    logging.info(f"Directorio de salida: {output_dir}")
    
    logging.info("=== Probando iterador básico ===")
    test_iterator(csv_files, output_dir)

    logging.info("=== Probando productor-consumidor ===")
    test_pipeline(csv_files, output_dir)