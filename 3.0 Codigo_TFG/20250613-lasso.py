# =============================================================================
# LASSO (con selección interna) + EVALUACIÓN RIGUROSA
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ── Scikit-learn --------------------------------------------------------------
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.feature_selection  import SelectKBest, f_regression
from sklearn.model_selection    import TimeSeriesSplit, cross_val_score
from sklearn.metrics            import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model       import LassoCV, LinearRegression
from sklearn.ensemble           import RandomForestRegressor

# ── Diagnóstico estadístico ---------------------------------------------------
import statsmodels.api          as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools  import durbin_watson

# =============================================================================
# 1. CONFIGURACIÓN GLOBAL
# =============================================================================
N_SPLITS         = 5         # para TimeSeriesSplit
K_FEATURES       = "all"     # N° de variables a seleccionar; "all" ⇒ sin recorte
BOOTSTRAP_ROUNDS = 100       # para estudiar estabilidad
RANDOM_STATE     = 42

# =============================================================================
# 2. UTILIDADES DE EVALUACIÓN
# =============================================================================
def metrics_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2"   : r2_score(y_true,   y_pred),
        "MAE"  : mean_absolute_error(y_true, y_pred),
        "RMSE" : np.sqrt(mse)
    }

def residual_diagnostics(y_true, y_pred):
    """Linealidad, homoscedasticidad y autocorrelación (pruebas clásicas)."""
    resid = y_true - y_pred
    # Breusch-Pagan
    lm, pval, *_ = het_breuschpagan(resid, sm.add_constant(y_pred))
    # Durbin-Watson
    dw = durbin_watson(resid)
    return {"BP_pvalue": pval, "DurbinWatson": dw}

# =============================================================================
# 3. CONSTRUCCIÓN DEL PIPELINE SIN FUGA DE INFORMACIÓN
# =============================================================================
def build_lasso_pipeline(k_features=K_FEATURES):
    return Pipeline([
        ("scaler"  , StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=k_features)),
        ("model"   , LassoCV(cv=TimeSeriesSplit(n_splits=N_SPLITS),
                             random_state=RANDOM_STATE,
                             max_iter=10000))
    ])

# =============================================================================
# 4. FUNCIÓN PRINCIPAL PARA UNA ESTACIÓN
# =============================================================================
def run_models_single_station(df_sta, station_id):
    """Entrena Lasso, dos baselines y un modelo no lineal; genera métricas + tests."""
    df_sta = df_sta.sort_values("fecha")          # ← CLAVE si los datos son temporales
    y      = df_sta["ca_entradas"].values
    X      = df_sta.drop(columns=["ca_entradas", "fecha"]).values

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.1 Split temporal (último 20 % como hold-out)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    split_idx   = int(len(df_sta) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4.2 MODELOS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    models = {
        "Mean_baseline"  : ("media", None),  # se gestiona aparte
        "OLS"            : ("linear", LinearRegression()),
        "Lasso"          : ("lasso",  build_lasso_pipeline()),
        "RandomForest"   : ("rf"   ,  RandomForestRegressor(
                                      n_estimators=300,
                                      random_state=RANDOM_STATE,
                                      n_jobs=-1))
    }
    results = []
    for name, (tag, model) in models.items():
        if tag == "media":                      # --- baseline trivial -------------
            y_pred = np.repeat(y_train.mean(), len(y_test))
            res    = metrics_dict(y_test, y_pred)
            res.update({"alpha": None, "features": None})
        else:                                   # --- modelos entrenables ----------
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res    = metrics_dict(y_test, y_pred)

            # α y variables seleccionadas sólo para Lasso
            if tag == "lasso":
                lasso      = model.named_steps["model"]
                selector   = model.named_steps["selector"]
                selected   = selector.get_support(indices=True)
                res.update({
                    "alpha"    : lasso.alpha_,
                    "features" : selected.tolist()
                })
            else:
                res.update({"alpha": None, "features": None})

        # Diagnóstico residual
        res.update(residual_diagnostics(y_test, y_pred))
        res.update({"model": name})
        results.append(res)

    results_df = pd.DataFrame(results)
    return results_df


# =============================================================================
# 5. ESTABILIDAD DE VARIABLES (BOOTSTRAP)
# =============================================================================
def variable_stability(df_sta):
    y = df_sta["ca_entradas"].values
    X = df_sta.drop(columns=["ca_entradas", "fecha"]).values
    n, p  = X.shape
    freq  = np.zeros(p)

    for _ in range(BOOTSTRAP_ROUNDS):
        idx      = np.random.choice(np.arange(n), size=n, replace=True)
        X_boot   = X[idx];   y_boot = y[idx]
        pipe     = build_lasso_pipeline()
        pipe.fit(X_boot, y_boot)
        selector = pipe.named_steps["selector"]
        mask     = selector.get_support() & (pipe.named_steps["model"].coef_ != 0)
        freq[mask] += 1

    stability = freq / BOOTSTRAP_ROUNDS
    return stability


# =============================================================================
# 6. EJECUCIÓN GLOBAL
# =============================================================================
if __name__ == "__main__":
    # 6.1 Cargar datos
    df = pd.read_csv(Path("data/dataset_metro.csv"), sep=";", decimal=",")
    # Asegúrate de que 'fecha' sea datetime si existe
    df["fecha"] = pd.to_datetime(dict(year = 2024, month = df["nu_mes"], day = df["nu_dia_mes"]))

    # 6.2 Diccionario por estación
    station_dfs = {
        est: data.drop(columns=["cod_est"]).reset_index(drop=True)
        for est, data in df.groupby("cod_est")
    }

    # 6.3 Iterar estaciones
    all_metrics = []
    for est, df_sta in station_dfs.items():
        try:
            res_df = run_models_single_station(df_sta, est)
            stab   = variable_stability(df_sta)
            res_df["station"] = est
            res_df["var_stability_mean"] = stab.mean()  # resumen rápido
            all_metrics.append(res_df)
        except Exception as e:
            print(f"⚠️  Estación {est}: {e}")

    final_df = pd.concat(all_metrics, ignore_index=True)
    final_df.to_csv("model_comparison_with_diagnostics.csv", index=False)
    print("✅ Archivo guardado: model_comparison_with_diagnostics.csv")
