import threading
import requests
import utils
from concurrent.futures import ThreadPoolExecutor
from utils import read_pokemons, ensure_dir, timeit
import pathlib
import typing as t
import time

# Almacenamiento local por hilo para la sesión
thread_local = threading.local()

# Contadores globales de éxito y fallo
success_count = 0
fail_count = 0

# Lista global para almacenar logs de descarga
download_logs: t.List[str] = []

# Bloqueo para modificar contadores desde múltiples hilos
lock = threading.Lock()

def main(output_dir: str, input_dir: str):
    """Download all Pokémon images from all CSVs in input_dir to output_dir."""
    global download_logs
    # Carpeta de salida
    output_path = pathlib.Path(output_dir)
    utils.ensure_dir(output_path)

    # Carpeta de depuración
    debug_path = pathlib.Path("debugs")
    utils.ensure_dir(debug_path)
    log_file = debug_path / "threading_download_log.txt"

    # Buscar todos los CSV en la carpeta input_dir
    input_path = pathlib.Path(input_dir)
    csv_files = list(input_path.glob("*.csv"))
    print(f"Encontrados {len(csv_files)} archivos CSV en {input_dir}")

    # Leer todas las filas de los CSV
    rows = list(utils.read_pokemons([str(f) for f in csv_files]))
    print(f"Se van a procesar {len(rows)} Pokémon")

    # Medir tiempo de ejecución
    start_time = time.perf_counter()
    download_all_pokemons(rows, output_dir) # Descargar todos los Pokémon concurrentemente
    duration = time.perf_counter() - start_time

    # Guardar log de descargas
    with open(log_file, "w") as f:
        f.write("\n".join(download_logs))
        f.write("\n")
        f.write(f"\nResumen:\nDescargados correctamente: {success_count}\nFallidos: {fail_count}\n")
        f.write(f"Tiempo total: {duration:.2f} segundos\n")

    print(f"\nDescarga completa en {duration:.2f} segundos")
    print(f"Pokémon descargados correctamente: {success_count}")
    print(f"Pokémon fallidos: {fail_count}")
    print(f"Log de depuración guardado en {log_file}")

# Funcion para descargar uno por uno
def download_one(row, output_dir):
    """Descargar un Pokémon y registrar el resultado."""
    global success_count, fail_count, download_logs

    session = get_session_for_thread()
    url = row["Sprite"]
    name = row["Pokemon"]
    type1 = row["Type1"]
    file_path = pathlib.Path(output_dir) / type1 / f"{name}.png"
    ensure_dir(file_path.parent)
    try:
        # Descarga la imagen
        bytes_ = utils.maybe_download_sprite(session, url)
        if bytes_:
            # Escribe la imagen en disco
            utils.write_binary(file_path, bytes_)
            with lock:
                success_count += 1
                download_logs.append(f"[OK] {name} ({type1})")
        else:
            with lock:
                fail_count += 1
                download_logs.append(f"[FAIL] {name} ({type1}) - No content")
    except Exception as e:
        with lock:
            fail_count += 1
            download_logs.append(f"[ERROR] {name} ({type1}) - {e}")

# Funcion para descargar en simultaneo con hilos
def download_all_pokemons(rows, output_dir):
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(lambda row: download_one(row, output_dir), rows)

# Funcion para obtener la sesion por hilo, reutilizando conexiones
def get_session_for_thread():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument("output_dir", help="directory to store the data")
    #parser.add_argument("inputs", nargs="+", help="list of files with metadata")
    #args = parser.parse_args()
    #utils.maybe_remove_dir(args.output_dir)
    #main(args.output_dir, args.inputs)
    # Carpeta de salida de las imagenes
    output_dir = "results/threading"
    input_dir = "data"
    main(output_dir, input_dir)