import pathlib
import requests
import time
import multiprocessing
import utils
import atexit

# Variables compartidas entre procesos
manager = multiprocessing.Manager() # Permite compartir listas y contadores entre procesos
# Contadores globales de éxito y fallo
success_count = manager.Value('i', 0)
fail_count = manager.Value('i', 0)
# El log se escribirá directamente en el archivo por cada proceso
log_file_path = pathlib.Path("debugs") / "multiprocessing_download_log.txt"

session: requests.Session  # Se inicializa por proceso


def init_process():
    """Inicializar la sesión HTTP por proceso."""
    global session
    session = requests.Session()
    atexit.register(session.close) # Se asegura de finalizar la sesión cuando el proceso termina


def download_one(row_and_output):
    """Descargar un Pokémon usando la sesión global del proceso."""
    #if session is None:

    row, output_dir = row_and_output
    url = row["Sprite"]
    name = row["Pokemon"]
    type1 = row["Type1"]
    file_path = pathlib.Path(output_dir) / type1 / f"{name}.png"
    utils.ensure_dir(file_path.parent)

    try:
        # Descarca con la sesion del proceso
        with session.get(url, timeout=10) as response:
            if response.status_code == 200:
                # Guarda el contenido binario en el archivo
                utils.write_binary(file_path, response.content)
                with success_count.get_lock():
                    success_count.value += 1
                log_line = f"[OK] {name} ({type1})"
            else:
                with fail_count.get_lock():
                    fail_count.value += 1
                log_line = f"[FAIL] {name} ({type1}) - No content"
    except Exception as e:
        with fail_count.get_lock():
            fail_count.value += 1
        log_line = f"[ERROR] {name} ({type1}) - {e}"
    # Escribir el log en modo append
    with open(log_file_path, "a") as f:
        f.write(log_line + "\n")


def main(output_dir: str, input_dir: str):
    output_path = pathlib.Path(output_dir)
    utils.ensure_dir(output_path)

    debug_path = pathlib.Path("debugs")
    utils.ensure_dir(debug_path)
    log_file = log_file_path
    # Limpiar el archivo de log antes de iniciar la corrida
    open(log_file, "w").close()

    # Leer todos los CSV
    input_path = pathlib.Path(input_dir)
    csv_files = list(input_path.glob("*.csv"))
    rows = list(utils.read_pokemons([str(f) for f in csv_files]))

    start_time = time.perf_counter()

    # Preparar lista de tuplas (row, output_dir) para pasar a los procesos
    row_output_pairs = [(row, output_dir) for row in rows] # porque executor.map solo acepta un argumento por iteración

    with multiprocessing.Pool(initializer=init_process) as executor:
        executor.starmap(download_one, row_output_pairs)
        executor.close()
        executor.join()

    duration = time.perf_counter() - start_time

    # Guardar resumen al final
    with open(log_file, "a") as f:
        f.write(f"\nResumen:\nDescargados correctamente: {success_count.value}\nFallidos: {fail_count.value}\n")
        f.write(f"Tiempo total: {duration:.2f} segundos\n")

    print(f"Descarga completa en {duration:.2f} segundos")
    print(f"Pokémon descargados correctamente: {success_count.value}")
    print(f"Pokémon fallidos: {fail_count.value}")
    print(f"Log de depuración guardado en {log_file}")



# Protege la ejecución de multiprocessing en Windows
if __name__ == "__main__":
    multiprocessing.freeze_support()
    output_dir = "results/multiprocessing"
    input_dir = "data"
    main(output_dir, input_dir)
