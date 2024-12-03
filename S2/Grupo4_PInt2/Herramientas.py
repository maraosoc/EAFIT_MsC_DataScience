import sqlite3
import pandas as pd
from fancyimpute import IterativeImputer
import numpy as np
import statsmodels.api as sm
import statsmodels
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from matplotlib.dates import DateFormatter

# skforecast

from skforecast.plot import set_dark_theme
from skforecast.sarimax import Sarimax 
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import backtesting_sarimax
from skforecast.model_selection  import grid_search_sarimax

from statsmodels.tsa.stattools import adfuller , kpss

# skforecast
from skforecast.model_selection import TimeSeriesFold
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import bayesian_search_forecaster

# modeling
import lightgbm
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('once')
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

class preprocesamiento:

    def recuperar ():
        # Conxión con la base de datos
        db_path = '../ETL/Data_Base.db'
        conn = sqlite3.connect(db_path)
        # query para obtener la tabla y cargarla en un dataframe
        query = "SELECT * FROM  procesamiento_proyecto_int"
        df = pd.read_sql_query(query, conn)

        # Tamaño del dataframe
        print(f'La base de datos tiene un tamaño de {df.shape[0]} filas y {df.shape[1]} columnas')
        
        return df
    
    def index_datetime(data):
        data = data.set_index('fecha_hora')
        data.index = pd.to_datetime(data.index)
        data = data.asfreq('H')
        return data

    def preprocesamiento_df(df):
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        #imputar todas las variables meteorologicas con MICE
        variables_total = df.iloc[:, 6:].columns # Todas las variables
        variables_meteo = df.iloc[:, 6:16].columns # Solo variables meteorologicas
        variables_meteo = list(variables_meteo)
        print("Las variables imputadas fueron: ",variables_meteo)
        df_copy = df.copy()
        df_copy['fecha_hora'] = pd.to_datetime(df_copy['fecha_hora'])
        date_index = df_copy['fecha_hora']
        df_copy = df_copy.drop(columns=[col for col in df_copy if col not in variables_meteo])
        mice_imputer = IterativeImputer()
        imputed_data = mice_imputer.fit_transform(df_copy)
        imputed_data = imputed_data.clip(min=0)
        imputed_df = pd.DataFrame(imputed_data, columns=df_copy.columns, index=date_index)
        imputed_df.reset_index(level=0, inplace=True)
        df['precipitacion'] = imputed_df['precipitacion'].values
        df['presion'] = imputed_df['presion'].values
        df['humedad'] = imputed_df['humedad'].values
        df['temperatura'] = imputed_df['temperatura'].values
        df['Velocidad_Prom'] = imputed_df['Velocidad_Prom'].values
        df['Velocidad_Max'] = imputed_df['Velocidad_Max'].values
        df['Direccion_Prom'] = imputed_df['Direccion_Prom'].values
        df['Direccion_Max'] = imputed_df['Direccion_Max'].values
        df['radiacion'] = imputed_df['radiacion'].values
        df['humedad_suelo'] = imputed_df['humedad_suelo'].values
        variables_modelo = list(variables_total)
        variables_modelo.remove('Velocidad_Max')
        variables_modelo.remove('Direccion_Max')
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        df["fecha"] = df["fecha_hora"].dt.strftime('%Y-%m-%d %H:%M')
        df["fecha"] = df["fecha"].astype(str)
        return df
    
    def crear_lags(df, columna, n_lags):
        """
        Crea variables de lag para una columna dada en un DataFrame.
        
        Parámetros:
        - df: DataFrame original.
        - columna: Nombre de la columna para la cual se crearán los lags.
        - n_lags: Número de lags a crear (por defecto 3).
        
        Retorna:
        - df con las columnas de lags añadidas.
        """
        for i in range(1, n_lags + 1):
            df[f'{columna}_lag_{i}'] = df[columna].shift(i)
        return df
    
    def preprocesar_datos_model(df, variables_lag, fecha_col='fecha', start='2023-07-01', end='2023-07-30'):
        # Crear el índice de tiempo
        df_index = pd.DataFrame(pd.date_range(start=df[fecha_col].min(), end=df[fecha_col].max(), freq='H').to_period(), columns=["fecha"])
        df_index['index_tiempo'] = df_index["fecha"]
        df_index['fecha'] = df_index["fecha"].astype(str)
        
        # Unir el índice al dataset original
        df = pd.DataFrame(df.join(df_index.set_index(["fecha"]), lsuffix="_x", rsuffix="_y", on=["fecha"], how="left"))
        df.sort_values(by=['index_tiempo'], inplace=True)
        df.set_index('index_tiempo', inplace=True)

        # Convertir algunas columnas a float
        df['mes'] = df['mes'].astype(float)
        df['semana_año'] = df['semana_año'].astype(float)
        df['hora'] = df['hora'].astype(float)
        
        # Crear los lags para las variables especificadas
        for var in variables_lag:
            df = preprocesamiento.crear_lags(df, var, n_lags=3)
        
        # Separar los datos en train y unseen (predicción)
        df_unseen = df[(df['fecha_hora'] >= start) & (df['fecha_hora'] < end)]
        df_train = df[(df['fecha_hora'] <start)]

        print(df_train.info())

        return df_train, df_unseen
    
class eda:
    def plot_time_series(data,titulo,color,linea,formato_fecha):
        if formato_fecha is not None:
            formato_fecha = '%Y-%m'
        else:
            formato_fecha
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        sns.set(style="whitegrid", font_scale=1.7)
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(data['fecha_hora'], data['humedad_suelo'], color=f'{color}',linewidth=int(linea))
        plt.title(f'{titulo}')
        plt.xlabel('Fecha')
        plt.ylabel('Humedad del suelo (%)')
        plt.gca().xaxis.set_major_formatter(DateFormatter(f'{formato_fecha}'))
        plt.legend()
        plt.xticks(rotation=0, ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()
        
    def d_time(data):
        data = data.set_index('fecha_hora')
        data.index = pd.to_datetime(data.index)
        data = data.asfreq('H')
        return data

    def descomposicion(df):
        df = df[['humedad_suelo']]
        df['fecha_hora'] = df.index
        df = df.asfreq('H')
        # Realizar la descomposición de la serie temporal
        res_decompose = seasonal_decompose(df['humedad_suelo'], model='additive', extrapolate_trend='freq', period=4)
        # Crear los subgráficos para visualizar los componentes
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 6), sharex=True)

        # Graficar los componentes de la descomposición
        res_decompose.observed.plot(ax=axs[0], color='navy')
        axs[0].set_title('Serie original', fontsize=14)

        res_decompose.trend.plot(ax=axs[1], color='navy')
        axs[1].set_title('Tendencia', fontsize=14)

        
        res_decompose.resid.plot(ax=axs[2], color='navy')
        axs[2].set_title('Residuales', fontsize=14)
        # Título general para el gráfico
        fig.suptitle('Descomposición de serie temporal', fontsize=14)
        # Ajuste de layout para evitar superposiciones
        fig.tight_layout()
        # Mostrar los gráficos
        plt.show()

        # Crear un gráfico adicional solo con la estacionalidad y hacer zoom en él
        fig_seasonal, ax_seasonal = plt.subplots(figsize=(9, 3))
        res_decompose.seasonal.plot(ax=ax_seasonal, color='navy')
        ax_seasonal.set_title('Zoom en la Estacionalidad', fontsize=15)
        zoom_start = '2022-06-01 00:00:00'
        zoom_end = '2022-06-01 23:00:00'
        ax_seasonal.set_xlim([pd.Timestamp(zoom_start), pd.Timestamp(zoom_end)])  # Usar fechas directamente
        # Configurar los ticks del eje X a una frecuencia horaria
        ax_seasonal.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Cada hora  # Formato de hora

        # Añadir una cuadrícula con frecuencia horaria
        ax_seasonal.grid(which='major', linestyle='--', linewidth=0.5, color='gray')

        fig_seasonal.tight_layout()

        # Mostrar el gráfico adicional
        plt.show()

        return
    
    def acf_pacf(data):
        data_acf = data
        data_acf['fecha_hora'] = data_acf.index
        data_acf = data_acf[['fecha_hora','humedad_suelo']]
        data_acf = eda.d_time(data_acf)
        # guarda en una variable el nombre de la columna
        variable_name = data_acf.columns[0]
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True)
        plot_acf(data_acf, lags=50, ax=ax[0], auto_ylims=True, color='navy')
        ax[0].set_title(f'Autocorrelación serie {variable_name}', fontsize=14)
        ax[0].set_ylabel('ACF')
        plot_pacf(data_acf, lags=50, ax=ax[1], auto_ylims=True, color='navy')
        ax[1].set_title(f'Autocorrelación parcial serie {variable_name}', fontsize=14)
        ax[1].set_ylabel('PACF')
        plt.show()
        return
    
    def ccf(df):

        return
    
    def graf_estacional(data):
        data['fecha_hora'] = data.index
        # Ensure 'fecha_hora' is datetime
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=True)
        axs = axs.ravel()
        
        # Users distribution by month
        data['month'] = data['fecha_hora'].dt.month
        data.boxplot(column='humedad_suelo', by='month', ax=axs[0], color='navy')
        data.groupby('month')['humedad_suelo'].median().plot(style='o-', linewidth=0.8, ax=axs[0])
        axs[0].set_ylabel('humedad suelo')
        axs[0].set_xlabel('mes')
        axs[0].set_title('Distribución de humedad suelo por mes', fontsize=15)

        # Users distribution by week day

        data['week_day'] = data['fecha_hora'].dt.day_of_week + 1
        data.boxplot(column='humedad_suelo', by='week_day', ax=axs[1], color='navy')
        data.groupby('week_day')['humedad_suelo'].median().plot(style='o-', linewidth=0.8, ax=axs[1])
        axs[1].set_ylabel('humedad suelo')
        axs[1].set_xlabel('día de semana')
        axs[1].set_title('Distribución humedad suelo por día de la semana.', fontsize=15)

        # Users distribution by the hour of the day
        data['hour_day'] = data['fecha_hora'].dt.hour + 1
        data.boxplot(column='humedad_suelo', by='hour_day', ax=axs[2], color='navy')
        data.groupby('hour_day')['humedad_suelo'].median().plot(style='o-', linewidth=0.8, ax=axs[2])
        axs[2].set_ylabel('humedad suelo')
        axs[2].set_xlabel('hora del dia')
        axs[2].set_title('Distribución de humedad suelo por hora del día', fontsize=15)

        # Users distribution by week day and hour of the day
        mean_day_hour = data.groupby(["week_day", "hour_day"])["humedad_suelo"].mean()
        mean_day_hour.plot(ax=axs[3], color='navy')
        axs[3].set(
            title       = "Media de la humedad suelo durante la semana",
            xticks      = [i * 24 for i in range(7)],
            xticklabels = ["lun", "mar", "mier", "jue", "vie", "sab", "dom"],
            xlabel      = "día semana",
            ylabel      = "Valor de humedad_suelo"
        )
        axs[3].title.set_size(15)

        fig.suptitle("Gráficas de estacionalidad", fontsize=15)
        fig.tight_layout()
        plt.show()

        return
    
    def plot_cross_correlation_subplots(data, variable_objetivo, variables, lags=50):
        """
        Genera subplots de correlación cruzada para una variable objetivo y un conjunto de variables en un DataFrame.

        Parámetros:
        - data (pd.DataFrame): DataFrame con las variables y una columna de fecha.
        - variable_objetivo (str): Nombre de la columna de la variable objetivo.
        - variables (list or dict): Lista o diccionario con los nombres de las variables a correlacionar.
        - lags (int): Número de lags a incluir en la correlación cruzada. Default: 50.
        """
        data['fecha_hora'] = data.index
        # Ensure 'fecha_hora' is datetime
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])

        # Calcular el número de filas necesarias para subplots
        n_vars = len(variables)
        n_cols = 2  # Número de columnas
        n_rows = -(-n_vars // n_cols)  # Calcular filas necesarias usando división entera redondeada hacia arriba

        # Crear los subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

        # Aplanar los ejes para indexarlos fácilmente (en caso de que sean 2D)
        axes = axes.flatten()

        # Iterar sobre las variables para generar las gráficas
        for i, variable in enumerate(variables):
            ax = axes[i]
            plot_ccf(data[variable_objetivo], data[variable], lags=lags, ax=ax, color='navy')
            ax.set_title(f'Correlación de {variable_objetivo} con {variable}', fontsize=15)
            ax.set_ylabel('CCF')
            ax.set_xlabel('Lags')

        # Eliminar subplots vacíos si hay más subplots que variables
        for j in range(len(variables), len(axes)):
            fig.delaxes(axes[j])

        plt.show()
    

class modelado:

    def search_space(trial):
        lags_grid = [1, 3, 12, 24]
        search_space  = {
            'n_estimators'    : trial.suggest_int('n_estimators', 50, 450, step=100),
            'max_depth'       : trial.suggest_int('max_depth', 3, 10, step=1),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 25, 500),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.5),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1, step=0.1),
            'max_bin'         : trial.suggest_int('max_bin', 50, 250, step=25),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1, step=0.1),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1, step=0.1),
            'lags'            : trial.suggest_categorical('lags', lags_grid)
        } 
        return search_space

    def tunning_lightgbm(data):
        data['fecha_hora'] = data.index
        data = data[['humedad_suelo', 'fecha_hora']]
        # definicion de conjuntos de entrenamiento, validacion y prueba
        fecha_inicio_val = '2022-07-01'
        fecha_fin_val = '2023-01-01'
        fecha_inicio_val = pd.to_datetime(fecha_inicio_val)
        fecha_fin_val = pd.to_datetime(fecha_fin_val)
        # Prepara los datos y separa en train y val
        df_train_sub = data[(data['fecha_hora'] < fecha_inicio_val)]
        data_train = data[(data['fecha_hora'] <= fecha_fin_val)]
        # Crear el forecaster
        window_features = RollingFeatures(stats=["mean"], window_sizes=24*30) # la media cambia en promedio cada mes
        forecaster = ForecasterRecursive(
                        regressor       = LGBMRegressor(random_state=15926, verbose=-1),
                        lags            = 1,
                        window_features = window_features
                    )

        cv = TimeSeriesFold(
                steps              = 24,
                initial_train_size = len(df_train_sub),
                refit              = False,
                fixed_train_size   = False,
        )
        
        resultados_busqueda, frozen_trial = bayesian_search_forecaster(
        forecaster    = forecaster,
        y             = data_train['humedad_suelo'], # Datos test no incluidos
        cv            = cv,
        search_space  = modelado.search_space,
        metric        = 'mean_absolute_error',
        n_trials      = 10, # Aumentar para una búsqueda más exhaustiva
        random_state  = 123,
        return_best   = True,
        n_jobs        = 'auto',
        verbose       = False,
        show_progress = True)

        return resultados_busqueda, forecaster

    

    def prep_split(data, fecha_col='fecha_hora', start='2023-01-01', end ='2023-07-30'):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        # separa los conjuntos en entenamiento y prueba
        df_test = data[(data['fecha_hora'] >= start) & (data['fecha_hora'] <= end)]
        df_train = data[(data['fecha_hora'] < start)]
        data = data[(data['fecha_hora'] <= end)]
        # establece la fecha como un índice de tipo datetime con frecuencia horaria para cada dataframe
        data = preprocesamiento.index_datetime(data)
        df_train = preprocesamiento.index_datetime(df_train)
        df_test = preprocesamiento.index_datetime(df_test)
        return df_train, df_test, data


    def modelo_sarima(data, s_order=24):
        data['fecha_hora'] = data.index
        data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
        data = data[['humedad_suelo', 'fecha_hora']]
        df_train, df_unseen, data = modelado.prep_split(data)
        
        # Sarima Backtest forecaster 
        
        forecaster = ForecasterSarimax(
                        regressor = Sarimax(
                                        order          = (1, 1, 1),
                                        seasonal_order =(1, 1, 1, s_order), 
                                        maxiter        = 10
                                    )
                    )
        
        cv = TimeSeriesFold(
                steps              = 24, # vamos a predecir el dia siguiente
                initial_train_size = len(df_train),
                refit              = True,
                fixed_train_size   = False,
        )

        metric, predictions = backtesting_sarimax(
                                forecaster            = forecaster,
                                y                     = data['humedad_suelo'],  
                                cv = cv,      
                                metric                = 'mean_absolute_error',
                                n_jobs                = "auto",
                                suppress_warnings_fit = True,
                                show_progress         = True
                            )
        # guarda en una variable las metricas 

        return metric, predictions
    
    def model_lightgbm(data, forecaster):
        data['fecha_hora'] = data.index
        data = data[['humedad_suelo', 'fecha_hora']]
        # Prepara los datos y separa en train y test
        fecha_inicio = '2023-01-01'
        fecha_fin = '2023-07-30'
        df_train, df_unseen, data = modelado.prep_split(data, start=fecha_inicio, end=fecha_fin)

        # Backtest del modelo LightGBM sin refit para mayor agilidad
        cv = TimeSeriesFold(
                steps              = 24,
                initial_train_size = len(df_train),
                refit              = True,
            )
        metrica, predictions = backtesting_forecaster(
                                    forecaster    = forecaster,
                                    y             = data['humedad_suelo'],
                                    cv            = cv,
                                    metric        = 'mean_absolute_error',
                                    n_jobs        = 'auto',
                                    verbose       = False,  # Cambiar a True para mostrar más información
                                    show_progress = True
                                )
        
        #evaluacion.plot_real_predit(df_unseen, predictions)
        # Visualizar resultados
        
        return metrica, predictions
    
    def modelo_sarimax(data, exogenas, numero_combinacion):
        data['fecha_hora'] = data.index
        df_train, df_unseen, data = modelado.prep_split(data)
        forecaster = ForecasterSarimax(
            regressor=Sarimax(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 24), # probar con 24 y 3,
                maxiter=10
            )
        )

        # Configuración de la validación cruzada
        cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(df_train),
            refit              = True,
            fixed_train_size   = False,
        )
        
        # Realizar backtesting
        metrics, predictions = backtesting_sarimax(
            forecaster=forecaster,
            y=data['humedad_suelo'],      # Serie objetivo
            exog=data[exogenas],         # Variables exógenas
            cv = cv,                         # Configuración de la validación cruzada
            metric='mean_absolute_error',    # Métrica de evaluación
            n_jobs="auto",                   # Procesamiento paralelo automático
            suppress_warnings_fit=True,      # Suprimir advertencias
            show_progress=True               # Progreso del backtesting
        )

        # Agregar columna de combinación a las predicciones
        predictions['numero_combinacion'] = numero_combinacion
        predictions['y_real'] = df_unseen['humedad_suelo']
        predictions['fecha'] = df_unseen.index

        return predictions
    
    def model_lightgbm_exog(data, exogenas, num_comb, params):
        data['fecha_hora'] = data.index
        target = 'humedad_suelo'
        include = [target]+exogenas+['fecha_hora']
        data = data[include]
        fecha_inicio = '2023-01-01'
        fecha_fin = '2023-07-30'
        data = data[(data['fecha_hora'] <= fecha_fin)]
        df_train, df_test, data = modelado.prep_split(data, start=fecha_inicio, end=fecha_fin)
        # Crear el forecaster
        window_features = RollingFeatures(stats=["mean"], window_sizes=24*30) # la media cambia en promedio cada mes
        forecaster = ForecasterRecursive(
                        regressor       = LGBMRegressor(**params),
                        lags            = 1,
                        window_features = window_features
                    )


        # Backtest del modelo LightGBM sin refit para mayor agilidad
        cv = TimeSeriesFold(
                steps              = 24,
                initial_train_size = len(df_train),
                refit              = True,
                fixed_train_size   = False,
            )

            
        metrica, predictions = backtesting_forecaster(
                                    forecaster    = forecaster,
                                    y             = data['humedad_suelo'],
                                    exog=data[exogenas], 
                                    cv            = cv,
                                    metric        = 'mean_absolute_error',
                                    n_jobs        = 'auto',
                                    verbose       = False,  # Cambiar a True para mostrar más información
                                    show_progress = True
                                )
        
        # Agregar columna de combinación a las predicciones
        predictions['num_comb'] = num_comb
        predictions['y_real'] = df_test['humedad_suelo']
        predictions['fecha'] = df_test.index
                
        return predictions
    

class evaluacion():

    def plot_inivariado(df_sarima3,df_test,df_sarima24,df_lgbm):
        df_sarima3['Unnamed: 0'] = pd.to_datetime(df_sarima3['Unnamed: 0'])
        df_sarima24['Unnamed: 0'] = pd.to_datetime(df_sarima24['Unnamed: 0'])
        df_lgbm['Unnamed: 0'] = pd.to_datetime(df_lgbm['Unnamed: 0'])
        df_test['fecha_hora'] = pd.to_datetime(df_test['fecha_hora'])
        # Definir la fecha límite para el filtro
        fecha_limite = '2023-07-30'
        fecha_limite = pd.to_datetime(fecha_limite)
        # Filtrar el DataFrame
        df_sarima3 = df_sarima3[df_sarima3['Unnamed: 0'] < fecha_limite]
        df_sarima24 = df_sarima24[df_sarima24['Unnamed: 0'] < fecha_limite]
        df_lgbm = df_lgbm[df_lgbm['Unnamed: 0'] < fecha_limite]
        sarima3 = df_sarima3.set_index('Unnamed: 0')['pred']
        sarima24 = df_sarima24.set_index('Unnamed: 0')['pred']
        test = df_test.set_index('fecha_hora')['humedad_suelo']
        lgbm = df_lgbm.set_index('Unnamed: 0')['pred']
        # Crear la figura y el gráfico
        plt.figure(figsize=(12, 6))
        # Graficar las tres series
        plt.plot(sarima3, label='Sarima 3', linestyle='-',color='orange',linewidth= 3)
        plt.plot(sarima24, label='Sarima 24', linestyle='--', color='green' ,linewidth= 3)
        plt.plot(test, label='Data testeo', color='navy',linewidth= 2)
        plt.plot(lgbm, label='LGBM', linestyle=':', color='aqua',linewidth= 3)
        # Personalizar el gráfico
        plt.title('Comparación modelos UNIVARIADOS', fontsize=15)
        plt.xlabel('Fecha (Año-mes)', fontsize=12)
        plt.ylabel('Humedad suelo (%)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # Mostrar el gráfico
        plt.show()
    
    def plot_real_predit(data,tipo_modelo):
        convinaciones= data['num_comb'].unique()
        for i in convinaciones:
            data['fecha'] = pd.to_datetime(data['fecha'])
            df_temp = data[data.num_comb==i]
            mae = mean_absolute_error(df_temp['y_real'], df_temp['pred'])
            mape = mean_absolute_percentage_error(df_temp['y_real'], df_temp['pred'])
            plt.figure(figsize=(10, 6))
            plt.plot(df_temp['fecha'], df_temp['y_real'], label='Datos Reales', color='navy')
            plt.plot(df_temp['fecha'], df_temp['pred'], label='Predicciones', color='orange', linestyle='dashed')
            plt.title(f'Pronostico humedad suelo Vs real {tipo_modelo} ({i}), (MAE = {mae:.2f}) y (MAPE = {mape:.2f}) ')
            plt.xlabel('Fecha')
            plt.ylabel('Humedad del suelo')
            plt.legend()
            plt.show()
        return 

    def plot_univariados(data,prediccion,metrica,tipo_modelo):
        data['fecha_hora'] = data.index
        df_train, df_unseen, data = modelado.prep_split(data)
        plt.figure(figsize=(15, 10))
        plt.plot(df_unseen.index, df_unseen['humedad_suelo'], label='Datos Reales', color='navy')
        plt.plot(prediccion.index, prediccion['pred'], label='Predicciones', color='orange', linestyle='dashed')
        plt.title(f'Pronostico humedad suelo Vs real {tipo_modelo}, (MAE = {metrica:.2f})')
        plt.xlabel('Fecha')
        plt.ylabel('Humedad del suelo')
        plt.legend()
        plt.show() 
        return 
  
