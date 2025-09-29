# Computer Vision Data Loader — Evaluación concurrente

Resumen
-------
Este repositorio compara dos enfoques para cargar imágenes de pokemones y medir su rendimiento:
- Iterador básico: lee CSVs y abre cada imagen secuencialmente.
- Productor-consumidor: productor que pone trabajos en una cola y consumidores que cargan imágenes mediante threads.


Estructura principal
--------------------
- `experiment.ipynb` — Notebook con el flujo de experimentos, carga de datos, ejecución de iterador básico y productor-consumidor, análisis y evaluación del tiempo de carga mediente gráficos.
- `src/loader.py` — Descarga concurrente de imágenes (solo una vez).
- `src/iterator.py` — Implementación del iterador básico y del productor-consumidor; guarda CSVs de tiempos de carga.
- `src/queue_tester.py` — Ejecuta pruebas con varios tamaños de cola y genera resumen y gráficas.
- `data/` CSVs con la información de los pokemones
- `output/` — Resultados: `pokemons/` (imágenes), `tiempos/` (en CSVs y subcarpeta para pruebas con diferentes tamaños de cola).

Requisitos y preparación (Windows / PowerShell)
----------------------------------------------
1. Crear y activar venv:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Registrar kernel en Jupyter:
   ```
   python -m ipykernel install --user --name venv --display-name "Python (venv)"
   ```
4. Establecer el directorio de trabajo:
   ```
   cd .\Mineria\computer-vision-data-loader\
   ```

Uso rápido (notebook)
---------------------
- Abrir `experiment.ipynb` en VS Code/Jupyter.
- Seleccionar kernel `Python (venv)`.
- Ejecutar celdas en orden:
  1. Descargar imágenes (solo la primera vez): llama a `src.loader.main`.
  2. Ejecutar `test_iterator(...)` para iterador básico (genera `output/tiempos/iterador_basico_tiempos.csv`).
  3. Ejecutar `test_pipeline(...)` para productor-consumidor (genera `output/tiempos/pipeline_tiempos.csv`).
  4. Probar tamaños de cola con `src.queue_tester.test_different_queue_sizes(...)` (genera subfolder `output/tiempos/queue_sizes/` con CSVs y gráficas).
  5. Ejecutar la función de comparación `compare_with_basic_iterator(...)` para la gráfica comparativa entre diferentes tamaños de cola y el iterador básico.


Recursos adicionales
---------------------

* https://realpython.com/python-concurrency/
* https://realpython.com/python-iterators-iterables/
* https://realpython.com/intro-to-python-threading/

**Repositorio de preparación con pruebas de diferentes enfoques concurrentes de descarga** https://github.com/maraosoc/EAFIT_MsC_DataScience/tree/main/Mineria/concurrent-downloads
