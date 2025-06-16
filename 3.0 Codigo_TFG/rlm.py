import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# === Funciones auxiliares ===

def corr_with_pvalues(df):
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
    return stats.shapiro(series)

def run_mlr_improved(df, target="ca_entradas", alpha_corr=0.05, min_abs_r=0.05,
                     test_size=0.2, random_state=42, min_nonzero=5, transform_log=False):
    """Ajuste de MLR con selección automática y filtrado de dummies escasas."""

    # ---- Filtrar dummies ultra‑escasas -----------------------
    if min_nonzero is not None and min_nonzero > 0:
        sparse_cols = [c for c in df.columns
                       if c != target and df[c].astype(bool).sum() < min_nonzero]
        if sparse_cols:
            df = df.drop(columns=sparse_cols)

    corr_m, pval_m = corr_with_pvalues(df)
    y_corr  = corr_m[target].abs()
    y_pvals = pval_m[target]
    keep = [
        col for col in df.columns
        if col != target and y_corr[col] >= min_abs_r and y_pvals[col] <= alpha_corr
    ]
    X = df[keep].copy()
    y = df[target].copy()

    if transform_log:
        X = X.applymap(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_c = sm.add_constant(X_train)
    X_test_c  = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train_c).fit(cov_type="HC3")
    y_pred = model.predict(X_test_c)

    mse_val = mean_squared_error(y_test, y_pred)
    metrics = {
        "R2":  r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mse_val),
        "AIC": model.aic,
        "BIC": model.bic
    }

    # ---- Test de heterocedasticidad (Bartlett) -------------
    fitted = model.fittedvalues
    resid  = model.resid
    groups = pd.qcut(fitted, 4, labels=False, duplicates="drop")
    group_lists = [resid[groups == g] for g in np.unique(groups)]
    bart_stat, bart_p = stats.bartlett(*group_lists)

    norm_stat, norm_p = normality_tests(model.resid)

    return {
        "model": model,
        "features": keep,
        "coeffs": model.params.to_dict(),
        **metrics,
        "Norm_p": norm_p,
        "Bartlett_p": bart_p,
    }

def full_process_improved(est_df_pair):
    est, df_sta = est_df_pair
    try:
        res = run_mlr_improved(df_sta)
        return {
            "cod_est": est,
            "n_obs": len(df_sta),
            "n_feat": len(res["features"]),
            "R2": res["R2"],
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "AIC": res["AIC"],
            "BIC": res["BIC"],
            "Norm_p": res["Norm_p"],
            "Bartlett_p": res["Bartlett_p"],
            "features": ", ".join(res["features"]),
            "coeffs": str(res["coeffs"]),
        }
    except Exception as e:
        return {"cod_est": est, "error": str(e)}

# === Función DEMO para una estación concreta ===

def demo_station_model(df_sta, station_code,
                       target="ca_entradas", alpha_corr=0.05, min_abs_r=0.05,
                       min_nonzero=5, transform_log=False):
    """Entrena el modelo de una estación y genera los gráficos básicos."""

    res = run_mlr_improved(df_sta, target=target, alpha_corr=alpha_corr,
                           min_abs_r=min_abs_r, min_nonzero=min_nonzero,
                           transform_log=transform_log)

    model    = res["model"]
    features = res["features"]

    # Predicciones sobre todos los datos de la estación
    X_full = sm.add_constant(df_sta[features])
    y_full = df_sta[target]
    y_pred = model.predict(X_full)
    resid  = y_full - y_pred

    # 1. Residuales vs predicción
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, color="red", lw=1)
    plt.xlabel("Predicción")
    plt.ylabel("Residuo")
    plt.title(f"Residuo vs Predicción - Est. {station_code}")
    plt.tight_layout()
    plt.show()

    # 2. Histograma de residuos
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, edgecolor="black")
    plt.xlabel("Residuo")
    plt.title(f"Histograma de residuos - Est. {station_code}")
    plt.tight_layout()
    plt.show()

    # 3. QQ plot
    plt.figure(figsize=(6, 4))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(f"QQ Plot - Est. {station_code}")
    plt.tight_layout()
    plt.show()

    # 4. Real vs Predicción
    plt.figure(figsize=(6, 4))
    plt.scatter(y_full, y_pred, alpha=0.6)
    lims = [min(y_full.min(), y_pred.min()), max(y_full.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Real")
    plt.ylabel("Predicción")
    plt.title(f"Real vs Predicción - Est. {station_code}")
    plt.tight_layout()
    plt.show()

    # 5. Coeficientes
    coeffs = {k: v for k, v in res["coeffs"].items() if k != "const"}
    plt.figure(figsize=(8, 4))
    plt.barh(list(coeffs.keys()), list(coeffs.values()))
    plt.xlabel("Coeficiente")
    plt.title(f"Coeficientes - Est. {station_code}")
    plt.tight_layout()
    plt.show()

    return res

# === Carga de datos y ejecución ===

file_path = Path("data/dataset_metro.csv")  # Ajusta la ruta si hace falta
df = pd.read_csv(file_path, sep=";", decimal=",")

station_dfs = {
    est: data.drop(columns=["cod_est"]).reset_index(drop=True)
    for est, data in df.groupby("cod_est")
}

with ThreadPoolExecutor() as pool:
    summaries_improved = list(pool.map(full_process_improved, station_dfs.items()))

results_df_improved = pd.DataFrame(summaries_improved)
results_df_improved.to_csv("MLR_summary_improved.csv", index=False)
print("✅ Resumen guardado en 'MLR_summary_improved.csv'")

# --- DEMO ---------------------------------------------------

demo_code = list(station_dfs.keys())[0]  # Cambia aquí si quieres otra estación
demo_station_model(station_dfs[demo_code], demo_code)
