import pathlib
import requests
import time
import multiprocessing
import utils
import atexit
from typing import Optional

session: Optional[requests.Session] = None  # Inicializar session como None para evitar errores


def init_process():
    """Inicializar la sesión HTTP por proceso."""
    global session
    session = requests.Session()
    atexit.register(session.close) # Se asegura de finalizar la sesión cuando el proceso termina


def download_one(row, output_dir):
    """Descargar un Pokémon usando la sesión global del proceso."""
    url = row["Sprite"]
    name = row["Pokemon"]
    type1 = row["Type1"]
    file_path = pathlib.Path(output_dir) / type1 / f"{name}.png"
    utils.ensure_dir(file_path.parent)

    # Crea de una sesión local temporal en caso de que no se inicialice correctamente
    _local_session = None
    _s = session
    if _s is None:
        _s = requests.Session()
        _local_session = _s

    try:
        # Descarca con la sesion del proceso
        with _s.get(url, timeout=10) as response:
            if response.status_code == 200:
                # Guarda el contenido binario en el archivo
                utils.write_binary(file_path, response.content)
                print(f"[OK] {name} ({type1})")
            else:
                print(f"[FAIL] {name} ({type1}) - No content")
    except Exception as e:
        print(f"[ERROR] {name} ({type1}) - {e}")

    finally:
        if _local_session is not None: # Cierra la sesión local si se creó
            _local_session.close()


def main(output_dir: str, input_dir: str):
    output_path = pathlib.Path(output_dir)
    utils.ensure_dir(output_path)

    # Leer todos los CSV
    input_path = pathlib.Path(input_dir)
    csv_files = list(input_path.glob("*.csv"))
    rows = list(utils.read_pokemons([str(f) for f in csv_files]))

    start_time = time.perf_counter()

    # Preparar lista de tuplas (row, output_dir) para pasar a los procesos
    row_output_pairs = [(row, output_dir) for row in rows] # porque executor.map solo acepta un argumento por iteración

    # Añadir un chunksize pequeño   
    chunksize = max(1, len(row_output_pairs) // ((multiprocessing.cpu_count() or 1) * 4))

    with multiprocessing.Pool(initializer=init_process) as executor:
        executor.starmap(download_one, row_output_pairs, chunksize=chunksize)
        executor.close()
        executor.join()

    duration = time.perf_counter() - start_time
    print(f"Descarga completa en {duration:.2f} segundos")




# Protege la ejecución de multiprocessing en Windows
if __name__ == "__main__":
    multiprocessing.freeze_support()
    output_dir = "results/multiprocessing"
    input_dir = "data"
    main(output_dir, input_dir)
