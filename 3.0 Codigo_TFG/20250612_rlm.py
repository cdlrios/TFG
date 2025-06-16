# ================================================================
# 0. Librerías
# ================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# ================================================================
# 1. Carga de datos
# ================================================================
file_path = Path("data/dataset_metro.csv")     # <-- AJUSTA AQUÍ
df = pd.read_csv(file_path, sep=";", decimal=",")             # ≈ 70 000 filas, 366×193

# ================================================================
# 2. Generar un DataFrame por estación y quitar 'cod_est'
# ================================================================
station_dfs = {
    est: data.drop(columns=["cod_est"]).reset_index(drop=True)
    for est, data in df.groupby("cod_est")
}

print(f"Se han creado {len(station_dfs)} DataFrames (uno por estación).")

# ================================================================
# 3. Elegir estaciones para la demo interactiva
# ================================================================
demo_stations = [1006, 708, 1002]   # <-- PON AQUÍ TUS 3 CÓDIGOS

# ================================================================
# 4. Funciones auxiliares
# ================================================================
def corr_with_pvalues(df):
    """Devuelve matriz de correlaciones y de p-values (Pearson)."""
    cols = df.columns
    corr_m = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_m = corr_m.copy()
    for col1, col2 in combinations(cols, 2):
        r, p = stats.pearsonr(df[col1], df[col2])
        corr_m.loc[col1, col2] = corr_m.loc[col2, col1] = r
        pval_m.loc[col1, col2] = pval_m.loc[col2, col1] = p
    np.fill_diagonal(corr_m.values, 1)
    np.fill_diagonal(pval_m.values, 0)
    return corr_m, pval_m

def normality_tests(series):
    """Shapiro (n<5 000) o D’Agostino.  Devuelve (stat, p-value)."""
    if len(series) < 5000:
        return stats.shapiro(series)
    else:
        return stats.normaltest(series)

def run_mlr(df, target="ca_entradas", alpha_corr=0.05, min_abs_r=0.05,
            test_size=0.2, random_state=42):
    """
    Entrena una regresión lineal múltiple y devuelve métricas + diagnóstico.
    • alpha_corr  -> nivel de significación para descartar variables
    • min_abs_r   -> umbral mínimo |r| para mantener la variable
    """
    # 1) Selección preliminar de variables por correlación
    corr_m, pval_m = corr_with_pvalues(df)
    y_corr  = corr_m[target].abs()
    y_pvals = pval_m[target]
    keep = [
        col for col in df.columns
        if col != target and y_corr[col] >= min_abs_r and y_pvals[col] <= alpha_corr
    ]
    X = df[keep]
    y = df[target]

    # 2) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3) Modelo (con constante) usando statsmodels para diagnóstico completo
    X_train_c = sm.add_constant(X_train)
    X_test_c  = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train_c).fit()

    # 4) Predicciones y métricas
    y_pred  = model.predict(X_test_c)

    mse_val = mean_squared_error(y_test, y_pred)   # ← sin 'squared'
    metrics = {
        "R2"  : r2_score(y_test, y_pred),
        "MAE" : mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mse_val),                  # ← raíz cuadrada manual
    }


    # 5) Normalidad de residuos
    norm_stat, norm_p = normality_tests(model.resid)

    # 6) Homocedasticidad (Breusch-Pagan)
    bp_test = het_breuschpagan(model.resid, X_train_c)
    bp_stat, bp_p = bp_test[0], bp_test[1]

    return {
        "model"   : model,
        "features": keep,
        **metrics,
        "Norm_p"  : norm_p,
        "BP_p"    : bp_p,
    }

def explore_station(df, cod):
    """Muestra estructura, correlaciones y resultados de tests para 1 estación."""
    print(f"\n\n========= Estación {cod} =========")
    print(f"Dimensiones: {df.shape}")
    df.info()
    print("\nDescripción estadística:")
    print(df.describe())

    # Correlaciones + significancia
    corr_m, pval_m = corr_with_pvalues(df)
    print("\nCorrelación con 'ca_entradas':")
    print(corr_m['ca_entradas'].sort_values(ascending=False))

    # Entrenar modelo y diagnósticos
    res = run_mlr(df)
    print("\n--- Métricas de prueba ---")
    print({k: round(v,4) for k,v in res.items() if k in ("R2","MAE","RMSE")})
    print(f"p-valor normalidad residuos  : {res['Norm_p']:.4f}")
    print(f"p-valor Breusch-Pagan (hetero): {res['BP_p']:.4f}")

# ================================================================
# 5. DEMO en vivo (3 estaciones)
# ================================================================
for est in demo_stations:
    explore_station(station_dfs[est], est)

# ================================================================
# 6. Proceso completo para las 193 estaciones
# ================================================================
def full_process(est_df_pair):
    est, df_sta = est_df_pair
    try:
        res = run_mlr(df_sta)
        return {
            "cod_est" : est,
            "n_obs"   : len(df_sta),
            "n_feat"  : len(res["features"]),
            "R2"      : res["R2"],
            "MAE"     : res["MAE"],
            "RMSE"    : res["RMSE"],
            "Norm_p"  : res["Norm_p"],
            "BP_p"    : res["BP_p"],
        }
    except Exception as e:
        return {"cod_est": est, "error": str(e)}

with ThreadPoolExecutor() as pool:
    summaries = list(pool.map(full_process, station_dfs.items()))

results_df = pd.DataFrame(summaries)
results_df.to_csv("MLR_summary_by_station.csv", index=False)
print("\nResumen guardado en 'MLR_summary_by_station.csv'")

print()
