import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, shapiro, bartlett
import numpy as np

# 1. Cargar dataset
df = pd.read_csv('data/dataset_final_1.csv', sep=';', decimal=',')
df.columns = df.columns.str.strip()  # Limpiar nombres de columnas

# 2. Crear diccionario de DataFrames por estación
dfs_estaciones = {
    cod: df_estacion.drop(columns='cod_est').reset_index(drop=True)
    for cod, df_estacion in df.groupby('cod_est')
}

# 3. Estaciones seleccionadas
codigos_estaciones = {
    'Bernabeu': 1006,
    'Sol': 112,
    'Begona': 1002,
    'ParqueAvenidas': 708
}

# 4. Crear figura de subplots
fig, axes = plt.subplots(
    nrows=len(codigos_estaciones),
    ncols=2,
    figsize=(20, 6 * len(codigos_estaciones))
)

# 5. Análisis para cada estación seleccionada
for i, (nombre, cod) in enumerate(codigos_estaciones.items()):
    print(f"\n===== Análisis estación: {nombre} (código {cod}) =====")
    df_est = dfs_estaciones[cod]

    # Información descriptiva
    print(df_est.head())
    print("\nNombres de las variables:")
    print(df_est.columns.tolist())
    print("\nClases de las variables:")
    print(df_est.dtypes)
    print("\nDimensiones del DataFrame:")
    print(df_est.shape)
    print("\nResumen estadístico:")
    print(df_est.describe())

    # Cálculo de correlaciones y p-valores con 'ca_entradas'
    correlaciones = {}
    pvalores = {}
    for col in df_est.columns:
        if col != 'ca_entradas':
            r, p = pearsonr(df_est[col], df_est['ca_entradas'])
            correlaciones[col] = r
            pvalores[col] = p

    # Filtrar variables con |r| > 0.1, p < 0.05, normalidad y homocedasticidad (Bartlett)
    variables_utiles = []
    for col in correlaciones:
        if abs(correlaciones[col]) > 0.1 and pvalores[col] < 0.05:
            p_shapiro_x = shapiro(df_est[col])[1]
            p_shapiro_y = shapiro(df_est['ca_entradas'])[1]
            try:
                p_bartlett = bartlett(df_est[col], df_est['ca_entradas'])[1]
            except ValueError:
                p_bartlett = 0  # en caso de varianza cero o problemas numéricos
            if p_shapiro_x > 0.05 and p_shapiro_y > 0.05 and p_bartlett > 0.05:
                variables_utiles.append(col)

    columnas_significativas = variables_utiles + ['ca_entradas']
    df_filtrado = df_est[columnas_significativas]

    # Cálculo de matrices de correlación y p-valores entre variables filtradas
    cols = df_filtrado.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            r, p = pearsonr(df_filtrado[col1], df_filtrado[col2])
            corr_matrix.loc[col1, col2] = r
            pval_matrix.loc[col1, col2] = p

    # Heatmap 1: correlación entre variables seleccionadas
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        annot_kws={"size": 4},
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f"Correlación ({nombre})", fontsize=12)
    axes[i, 0].tick_params(labelsize=6)
    axes[i, 0].tick_params(axis='x', rotation=45)

    # Heatmap 2: correlaciones significativas (p < 0.05)
    mask_insignificante = pval_matrix >= 0.05
    sns.heatmap(
        corr_matrix,
        mask=mask_insignificante,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        annot_kws={"size": 4},
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"Significativas (p<0.05) - {nombre}", fontsize=12)
    axes[i, 1].tick_params(labelsize=6)
    axes[i, 1].tick_params(axis='x', rotation=45)

# Mostrar todos los heatmaps juntos
plt.tight_layout()
plt.show()
