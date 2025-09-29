"""
Módulo para evaluar el rendimiento de diferentes tamaños de cola en el data loader
con enfoque productor-consumidor.
"""
import os
import sys
import time
import io
import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.iterator import test_pipeline
from src.utils import ensure_dir


def test_different_queue_sizes(csv_files, img_dir, queue_sizes=[5, 10, 30, 50, 100]):
    """
    Prueba el pipeline productor-consumidor con diferentes tamaños de cola
    y guarda los resultados en CSV separados.
    
    Args:
        csv_files (list): Lista de archivos CSV con datos de pokémon
        img_dir (str): Directorio donde están las imágenes
        queue_sizes (list): Lista de tamaños de cola a probar
    
    Returns:
        pandas.DataFrame: DataFrame con los resultados de las pruebas
    """
    # Crear subdirectorio para los resultados de diferentes tamaños de cola
    queue_times_dir = os.path.join('output', 'tiempos', 'queue_sizes')
    ensure_dir(queue_times_dir)

    # Configurar logging para suprimir salidas
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    
    results = []
    
    print(f"Probando {len(queue_sizes)} tamaños de cola diferentes: {queue_sizes}")
    
    # Probar cada tamaño de cola
    for size in queue_sizes:
        # Redirigir stdout para evitar que se muestren mensajes
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            print(f"\n--- Probando cola de tamaño {size} ---")
            pipeline_times_file = os.path.join(queue_times_dir, f'pipeline_queue_size_{size}.csv')
            
            # Registrar tiempo total para esta configuración
            start_time = time.perf_counter()
            count = test_pipeline(csv_files, img_dir, maxsize=size, csv_output=pipeline_times_file, show_first=False)
            total_time = time.perf_counter() - start_time
            
            # Cargar los tiempos del CSV que generó test_pipeline
            df = pd.read_csv(pipeline_times_file)
            
            # Agregar estadísticas a los resultados
            results.append({
                'queue_size': size,
                'processed_images': count,
                'total_time': total_time,
                'avg_time': df['duration'].astype(float).mean() if 'duration' in df.columns else 0,
                'median_time': df['duration'].astype(float).median() if 'duration' in df.columns else 0,
                'min_time': df['duration'].astype(float).min() if 'duration' in df.columns else 0,
                'max_time': df['duration'].astype(float).max() if 'duration' in df.columns else 0
            })
        finally:
            # Restaurar stdout
            sys.stdout = original_stdout   
            print(f"Cola tamaño {size}: {count} imágenes procesadas en {total_time:.2f} segundos")

    # Restaurar nivel de logging original
    logging.getLogger().setLevel(original_level)
    
    # Crear DataFrame con resultados y guardar en CSV
    results_df = pd.DataFrame(results)
    summary_file = os.path.join(queue_times_dir, 'queue_sizes_summary.csv')
    results_df.to_csv(summary_file, index=False)
    print(f"\nResumen guardado en {summary_file}")
    
    return results_df


def plot_queue_performance(results_df, save_path=None):
    """
    Visualiza los resultados de las pruebas de diferentes tamaños de cola.
    
    Args:
        results_df (pandas.DataFrame): DataFrame con los resultados de las pruebas
        save_path (str, optional): Ruta donde guardar la imagen del gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['queue_size'], results_df['total_time'], 's--', linewidth=2, label='Tiempo total')
    plt.xlabel('Tamaño de cola')
    plt.ylabel('Tiempo total(segundos)')
    plt.title('Comparación de rendimiento según tamaño de cola')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def compare_with_basic_iterator(queue_performance, times_file, queue_sizes, save_path=None):
    """
    Crea gráficos comparativos entre el iterador básico y diferentes tamaños de cola.
    
    Args:
        queue_performance (pandas.DataFrame): DataFrame con los resultados de diferentes tamaños de cola
        times_file (str): Ruta al CSV con los tiempos del iterador básico
        queue_sizes (list): Lista de tamaños de cola utilizados
        save_path (str, optional): Ruta donde guardar la imagen del gráfico
    
    Returns:
        pandas.DataFrame: DataFrame con el resumen comparativo
    """
    # Cargar datos del iterador básico
    basic_iterator_df = pd.read_csv(times_file)
    basic_total_time = basic_iterator_df['duration'].astype(float).sum()
    basic_avg_time = basic_iterator_df['duration'].astype(float).mean()
    
    # Crear figura para los gráficos
    plt.figure(figsize=(12, 8))

    # Comparar tiempo total
    plt.subplot(2, 1, 1)
    plt.bar(['Iterativo'] + [f'Cola={size}' for size in queue_sizes], 
            [basic_total_time] + list(queue_performance['total_time']),
            color=['gray'] + ['blue']*len(queue_sizes))
    plt.title('Comparación de tiempo total de procesamiento')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(axis='y')
    plt.xticks(rotation=45)

    # Comparar tiempo promedio
    plt.subplot(2, 1, 2)
    plt.bar(['Iterativo'] + [f'Cola={size}' for size in queue_sizes], 
            [basic_avg_time] + list(queue_performance['avg_time']),
            color=['gray'] + ['green']*len(queue_sizes))
    plt.title('Comparación de tiempo promedio por imagen')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(axis='y')
    plt.xticks(rotation=45)

    plt.tight_layout()
    
    # Guardar gráfico si se especificó una ruta
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    # Crear tabla de resumen
    summary_df = pd.DataFrame({
        'Método': ['Iterativo'] + [f'Cola={size}' for size in queue_sizes],
        'Tiempo total (s)': [basic_total_time] + list(queue_performance['total_time']),
        'Tiempo promedio (s)': [basic_avg_time] + list(queue_performance['avg_time']),
        'Imágenes': [len(basic_iterator_df)] + list(queue_performance['processed_images'])
    })
    
    return summary_df


if __name__ == "__main__":
    import glob
    
    # Ejemplo de uso del módulo
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    img_dir = os.path.join(base_dir, 'output', 'pokemons')
    times_dir = os.path.join(base_dir, 'output', 'tiempos')
    times_file = os.path.join(times_dir, 'iterador_basico_tiempos.csv')
    
    # Asegurar que existan los directorios
    ensure_dir(img_dir)
    ensure_dir(times_dir)
    
    # Probar diferentes tamaños de cola
    queue_sizes = [5, 10, 30, 50, 100]
    results = test_different_queue_sizes(csv_files, img_dir, queue_sizes)
    
    # Visualizar resultados
    plot_path = os.path.join(times_dir, 'queue_sizes', 'performance_plot.png')
    plot_queue_performance(results, save_path=plot_path)
    
    # Comparar con el iterador básico
    if os.path.exists(times_file):
        comparison_path = os.path.join(times_dir, 'queue_sizes', 'comparison_plot.png')
        summary = compare_with_basic_iterator(results, times_file, queue_sizes, save_path=comparison_path)
        print("\nComparación con el iterador básico:")
        print(summary)