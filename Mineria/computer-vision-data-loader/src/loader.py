import threading
import queue
import pathlib
import requests
import csv
import time
import logging
from typing import List

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

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# ---------------- PRODUCTOR ----------------
def producer(csv_files: List[str]):
    """
    Lee los CSV de pokemones y coloca en la cola los datos de descarga.
    """
    try:
        for filepath in csv_files:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["Type1"] = row["Type1"].lower()
                    row["Pokemon"] = row["Pokemon"].lower()
                    pipeline.put(row)  # bloquea si la cola está llena
                    logging.info(f"Producer: added {row['Pokemon']} to queue")
    finally:
        logging.info("Producer: finished producing, setting event")
        event.set()  # señal para que los consumidores terminen cuando la cola esté vacía

# ---------------- CONSUMIDOR ----------------
def consumer(output_dir: str, downloaded: list, failed: list):
    """
    Descarga los sprites de Pokémon y los guarda en output_dir.
    Guarda nombres de pokemones descargados y fallidos en listas compartidas.
    """
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
                    logging.info(f"Consumer: downloaded {name}")
                else:
                    failed.append(name)
                    logging.warning(f"Consumer: failed {name}, status {response.status_code}")
        except Exception as e:
            failed.append(name)
            logging.error(f"Consumer: error downloading {name} -> {e}")

# ---------------- MAIN ----------------
def main(csv_files: List[str], output_dir: str, num_consumers: int = 4):
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
    logging.info(f"Downloaded {len(downloaded)} pokemons successfully")
    logging.info(f"Failed to download {len(failed)} pokemons")
    logging.info(f"Total duration: {duration:.2f} seconds")

    # Guardar log de resultados
    debug_file = pathlib.Path("debugs/pokemon_download_log.txt")
    ensure_dir(debug_file.parent)
    with open(debug_file, "w") as f:
        f.write(f"Downloaded ({len(downloaded)}):\n")
        f.write("\n".join(downloaded))
        f.write("\n\nFailed ({len(failed)}):\n")
        f.write("\n".join(failed))

# ---------------- EJEMPLO DE USO ----------------
if __name__ == "__main__":
    import glob
    csv_files = glob.glob("../data/*.csv")
    main(csv_files, output_dir="../output/pokemons")
