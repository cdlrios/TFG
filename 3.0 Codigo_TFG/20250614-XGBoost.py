# xgb_full_power_threads.py
# ==============================================================
# XGBoost con 600 árboles por estación (paralelismo por hilos)
# ==============================================================

import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def train_station_xgb(est, df_sta, min_abs_r: float = 0.0):
    """Entrena XGBRegressor en una estación y devuelve métricas."""
    y = df_sta["ca_entradas"]
    X = df_sta.drop(columns=["ca_entradas"])

    # Filtro opcional (desactívalo dejando min_abs_r=0.0)
    if min_abs_r > 0:
        corr = X.corrwith(y).abs()
        X = X[corr[corr >= min_abs_r].index]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",   # CPU rápido
        n_jobs=6,             # usa los 6 núcleos
        random_state=42
    )

    model.fit(X_tr, y_tr, verbose=False)
    y_pred = model.predict(X_te)

    rmse = mean_squared_error(y_te, y_pred) ** 0.5

    return {
        "cod_est": est,
        "R2"    : r2_score(y_te, y_pred),
        "MAE"   : mean_absolute_error(y_te, y_pred),
        "RMSE"  : rmse,
        "n_feat": X.shape[1]
    }


def main():
    # 1. Cargar datos
    df = pd.read_csv(Path("data/dataset_metro.csv"), sep=";", decimal=",")
    station_dfs = {
        est: data.drop(columns=["cod_est"]).reset_index(drop=True)
        for est, data in df.groupby("cod_est")
    }

    # 2. Paralelizar con backend "threading" (misma librería en todos los hilos)
    results = Parallel(n_jobs=6, backend="threading")(
        delayed(train_station_xgb)(est, df_sta) for est, df_sta in station_dfs.items()
    )

    # 3. Guardar
    pd.DataFrame(results).to_csv("xgb_summary_full_power.csv", index=False)
    print("✅  Resultados guardados en xgb_summary_full_power.csv")


if __name__ == "__main__":
    main()
