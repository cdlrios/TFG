import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import combinations

# ===============================
# Función de demo para una estación
# ===============================
def lasso_demo(df_sta, station_id, min_abs_r=0.05):
    print(f"\n📍 Estación {station_id} - {df_sta.shape[0]} observaciones")

    y = df_sta["ca_entradas"]
    X = df_sta.drop(columns=["ca_entradas"])
    corr = X.corrwith(y).abs()
    keep_cols = corr[corr >= min_abs_r].index.tolist()
    print(f"🔍 Variables seleccionadas por |r| > {min_abs_r}: {len(keep_cols)}")

    X_sel = X[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    lasso = pipe.named_steps["lasso"]
    used_features = np.array(keep_cols)[lasso.coef_ != 0]

    mse_val = mean_squared_error(y_test, y_pred)   # ← sin 'squared'
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mse_val),
        "n_features_used": len(used_features),
        "alpha": lasso.alpha_,
        "features": list(used_features)
    }

    print(f"✅ R²: {metrics['R2']:.4f} | MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f}")
    print(f"🧮 Alpha óptimo: {metrics['alpha']:.4f}")
    print(f"📌 Variables seleccionadas: {metrics['features']}")

    return metrics


# ===============================
# Lasso para todas las estaciones
# ===============================
def run_lasso_all_stations(station_dfs, min_abs_r=0.05):
    results = []

    for est, df_sta in station_dfs.items():
        try:
            y = df_sta["ca_entradas"]
            X = df_sta.drop(columns=["ca_entradas"])
            corr = X.corrwith(y).abs()
            keep_cols = corr[corr >= min_abs_r].index.tolist()
            X_sel = X[keep_cols]

            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            lasso = pipe.named_steps["lasso"]
            used_features = np.array(keep_cols)[lasso.coef_ != 0]
            res_mse_val = mean_squared_error(y_test, y_pred)   # ← sin 'squared'
            res = {
                "cod_est": est,
                "n_obs": len(df_sta),
                "n_feat_init": len(keep_cols),
                "n_feat_lasso": len(used_features),
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(res_mse_val),
                "alpha": lasso.alpha_,
                "features": ", ".join(used_features)
            }
            results.append(res)
        except Exception as e:
            results.append({"cod_est": est, "error": str(e)})

    return pd.DataFrame(results)


# ===============================
# EJECUCIÓN PRINCIPAL
# ===============================
if __name__ == "__main__":
    # Cargar datos
    file_path = Path("data/dataset_metro.csv")
    df = pd.read_csv(file_path, sep=";", decimal=",")

    # Crear diccionario por estación
    station_dfs = {
        est: data.drop(columns=["cod_est"]).reset_index(drop=True)
        for est, data in df.groupby("cod_est")
    }

    # DEMO: aplicar a una estación
    demo_metrics = lasso_demo(station_dfs[1006], 1006)

    # PROCESO GLOBAL: aplicar a todas las estaciones
    df_lasso_all = run_lasso_all_stations(station_dfs)
    df_lasso_all.to_csv("lasso_summary_all_stations.csv", index=False)
    print("\n📁 Resultados guardados en: lasso_summary_all_stations.csv")
