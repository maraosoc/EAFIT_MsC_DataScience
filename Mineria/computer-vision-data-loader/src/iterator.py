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

def test_iterator(csv_files, output_dir, csv_output="iterador_basico_tiempos.csv"):
    """Prueba el iterador básico y guarda los tiempos en CSV"""
    iterator = rows_from_csv(csv_files, output_dir)
    
    # Preparar archivo CSV para guardar tiempos
    with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'duration', 'type1'])  # Encabezados
        
        try:
            count = 0
            for row in iterator:  # Ya tienes el row del iterador
                t0 = time.perf_counter()
                # Simular procesamiento de la imagen (ya está cargada)
                _ = row.image.size  # Acceder a la imagen
                duration = time.perf_counter() - t0

                # Guardar en CSV (dentro del bloque with)
                writer.writerow([row.name, f"{duration:.5f}", row.type1])
                logging.info(f"Iterador básico: imagen ({row.name}) cargada en {duration:.5f}s")

                # Solo mostrar la primera imagen procesada
                if count == 0:
                    plt.title(row.name)
                    plt.imshow(row.image)
                    plt.show()

                count += 1
            
        except StopIteration:
            logging.error("No hay imágenes válidas en la ruta.")
            return None
    
    logging.info(f"Tiempos del iterador básico guardados en {csv_output}")
    return count

# ------------------ PRODUCTOR Y CONSUMIDOR ------------------
def test_pipeline(csv_files, output_dir, maxsize=20, csv_output="pipeline_tiempos.csv"):
    """Prueba el pipeline productor-consumidor y guarda los tiempos en CSV"""
    # Cola compartida entre productor y consumidor
    pipeline = queue.Queue(maxsize=maxsize)
    event = threading.Event()

    # Lista thread-safe para guardar los tiempos
    times_data = []
    times_lock = threading.Lock()

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
        processed_count = 0

        while not event.is_set() or not pipeline.empty():
            try:
                row = pipeline.get(timeout=0.1)
                t0 = time.perf_counter()
                # Simular procesamiento de la imagen
                _ = row.image.size  # Acceder a la imagen
                duration = time.perf_counter() - t0
                
                processed_count += 1
                logging.info(f"Productor-Consumidor: imagen ({row.name}) cargada en {duration:.5f}s")

                # Guardar datos de forma thread-safe
                with times_lock:
                    times_data.append([row.name, f"{duration:.5f}", row.type1])

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
    
    # Guardar los tiempos en un archivo CSV
    with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'duration', 'type1'])  # Encabezados
        writer.writerows(times_data)
    
    logging.info(f"Tiempos del pipeline guardados en {csv_output}")
    logging.info("Procesamiento completo.")

    return len(times_data)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    output_dir = os.path.join(base_dir, 'output','pokemons')
    ensure_dir(output_dir)
    output_csv_path = os.path.join(base_dir, 'output','tiempos')
    ensure_dir(output_csv_path)
    output_csv_iterator = os.path.join(output_csv_path, 'iterador_basico_tiempos.csv')
    output_csv_pipeline = os.path.join(output_csv_path, 'pipeline_tiempos.csv')

    logging.info(f"Archivos CSV encontrados: {len(csv_files)}")
    logging.info(f"Directorio de salida: {output_dir}")
    
    logging.info("=== Probando iterador básico ===")
    test_iterator(csv_files, output_dir, output_csv_iterator)

    logging.info("=== Probando productor-consumidor ===")
    test_pipeline(csv_files, output_dir, maxsize=20, csv_output=output_csv_pipeline)