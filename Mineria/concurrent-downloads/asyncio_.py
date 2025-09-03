import asyncio
import pathlib
import time
import aiohttp
import utils

DEBUG_DIR = pathlib.Path("debugs")
utils.ensure_dir(DEBUG_DIR)
LOG_FILE = DEBUG_DIR / "async_download_log.txt"


async def download_one(row, output_dir, session, success_log, fail_log):
    """Descargar un Pokémon asíncronamente y registrar resultados."""
    url = row["Sprite"]
    name = row["Pokemon"]
    type1 = row["Type1"]
    file_path = pathlib.Path(output_dir) / type1 / f"{name}.png"
    utils.ensure_dir(file_path.parent)

    try:
        # Crea una sesión HTTP compartida entre todas las descargas
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read() # para no bloquear el loop principal
                utils.write_binary(file_path, content)
                success_log.append(f"[OK] {name} ({type1})")
            else:
                fail_log.append(f"[FAIL] {name} ({type1}) - Status {response.status}")
    except Exception as e:
        fail_log.append(f"[ERROR] {name} ({type1}) - {e}")


async def download_all_pokemons(rows, output_dir):
    """Descargar todos los Pokémon concurrentemente usando asyncio y aiohttp."""
    # Listas para almacenar logs de éxito y fallo
    success_log = []
    fail_log = []

    async with aiohttp.ClientSession() as session:
        tasks = [download_one(row, output_dir, session, success_log, fail_log) for row in rows]
        # Llama a gather async. para ejecutar todas las tareas simultáneamente
        await asyncio.gather(*tasks, return_exceptions=True) # evita que una excepción detenga todo

    # Guardar log
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(success_log + fail_log))
        f.write(f"\nResumen:\nDescargados correctamente: {len(success_log)}\nFallidos: {len(fail_log)}\n")

    print(f"Pokémon descargados correctamente: {len(success_log)}")
    print(f"Pokémon fallidos: {len(fail_log)}")


async def main(output_dir: str, input_dir: str):
    utils.ensure_dir(output_dir)
    input_path = pathlib.Path(input_dir)
    csv_files = list(input_path.glob("*.csv"))
    rows = list(utils.read_pokemons([str(f) for f in csv_files]))

    start_time = time.perf_counter()
    # Llama a la funcion para descargar todos los Pokémon
    await download_all_pokemons(rows, output_dir)
    duration = time.perf_counter() - start_time
    # Guardar la duración total de la descarga
    with open(LOG_FILE, "a") as f:
        f.write(f"Tiempo total: {duration:.2f} segundos\n")
    print(f"Descarga completa en {duration:.2f} segundos")


if __name__ == "__main__":
    output_dir = "results/asyncio"
    input_dir = "data"
    utils.maybe_remove_dir(output_dir)
    asyncio.run(main(output_dir, input_dir))
