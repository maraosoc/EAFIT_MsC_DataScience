import threading
import queue
import pathlib
import requests
import csv
import time
import logging
from typing import List
import glob
import os
from utils import ensure_dir

# Configuración de logging
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)

# Cola compartida entre productor y consumidor
pipeline = queue.Queue(maxsize=20)  # ajustable según memoria y velocidad
event = threading.Event()

# Thread-local para mantener una sesión por hilo
thread_local = threading.local()

def get_session_for_thread():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


# ---------------- PRODUCTOR ----------------
def producer(csv_files: List[str]):
    """
    Lee los CSV de pokemones y coloca en la cola los datos de descarga.
    """
    try:
        for filepath in csv_files:
            # Añadir encoding para evitar errores con caracteres especiales
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["Type1"] = row["Type1"].lower()
                    row["Pokemon"] = row["Pokemon"].lower()
                    pipeline.put(row)  # bloquea si la cola está llena
                    logging.info(f"Productor: {row['Pokemon']} añadido a la fila")
    finally:
        logging.info("Productor: finished producing, setting event")
        event.set()  # señal para que los consumidores terminen cuando la cola esté vacía

# ---------------- CONSUMIDOR ----------------
def consumer(output_dir: str, downloaded: list, failed: list):
    """
    Descarga los sprites de Pokémon y los guarda en output_dir.
    Guarda nombres de pokemones descargados y fallidos en listas compartidas.
    """
    try:
        while not event.is_set() or not pipeline.empty():
            try:
                row = pipeline.get(timeout=0.1)  # espera por un item si está vacía
            except queue.Empty:
                continue  # sigue si la cola está temporalmente vacía

            url = row["Sprite"]
            name = row["Pokemon"]
            type1 = row["Type1"]

            file_path = pathlib.Path(output_dir) / type1 / f"{name}.png"
            ensure_dir(file_path.parent)
            

            try:
                session = get_session_for_thread()
                with session.get(url) as response:
                    if response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        downloaded.append(name)
                        logging.info(f"Consumer [{threading.current_thread().name}]: downloaded {name}")
                    else:
                        failed.append(name)
                        logging.warning(f"Consumer [{threading.current_thread().name}]: failed {name}, status {response.status_code}")
            except Exception as e:
                failed.append(name)
                logging.error(f"Consumer [{threading.current_thread().name}]: error downloading {name} -> {e}")
    finally:
        # Cerrar la sesión asociada al hilo si se creó
        if hasattr(thread_local, "session"):
            try:
                thread_local.session.close()
            except Exception:
                pass


# ---------------- MAIN ----------------
def main(csv_files: List[str], output_dir: str, num_consumers: int = 8):
    downloaded = []
    failed = []

    start_time = time.perf_counter()

    # Lanzar productor
    producer_thread = threading.Thread(target=producer, args=(csv_files,))
    producer_thread.start()

    # Lanzar consumidores
    consumer_threads = [
        threading.Thread(target=consumer, args=(output_dir, downloaded, failed))
        for _ in range(num_consumers)
    ]
    for t in consumer_threads:
        t.start()

    # Esperar que termine todo
    producer_thread.join()
    for t in consumer_threads:
        t.join()

    duration = time.perf_counter() - start_time
    logging.info(f"{len(downloaded)} pokemones descargados con éxito")
    logging.info(f"Falló la descarga de {len(failed)} pokemones")
    logging.info(f"Duración total: {duration:.5f} segundos")

    # Guardar log de resultados
    debug_file = pathlib.Path("debugs/pokemon_download_log.txt")
    ensure_dir(debug_file.parent)
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"Descargados ({len(downloaded)}):\n")
        f.write("\n".join(downloaded))
        f.write(f"\n\nFallidos ({len(failed)}):\n")
        f.write("\n".join(failed))

# ---------------- EJEMPLO DE USO ----------------
if __name__ == "__main__":
    # Obtener ruta absoluta de la carpeta actual del script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    output_dir = os.path.join(base_dir, 'output', 'pokemons')
    ensure_dir(output_dir)

    main(csv_files, output_dir=output_dir, num_consumers=4)
