import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===============================
# Random Forest para todas las estaciones
# ===============================
def run_rf_all_stations(station_dfs, min_abs_r=0.05):
    summaries = []

    for est, df_sta in station_dfs.items():
        try:
            y = df_sta["ca_entradas"]
            X = df_sta.drop(columns=["ca_entradas"])

            # (opcional) selección preliminar por correlación
            corr = X.corrwith(y).abs()
            keep_cols = corr[corr >= min_abs_r].index.tolist()
            X_sel = X[keep_cols]

            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=0.2, random_state=42
            )

            rf = RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            mse  = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5

            summaries.append({
                "cod_est": est,
                "RMSE"   : rmse,
                "MAE"    : mean_absolute_error(y_test, y_pred),
                "R2"     : r2_score(y_test, y_pred),
                "n_feat" : len(keep_cols)
            })
        except Exception as e:
            summaries.append({"cod_est": est, "error": str(e)})

    return pd.DataFrame(summaries)

# ===============================
# Carga y ejecución
# ===============================
if __name__ == "__main__":
    file_path = Path("data/dataset_metro.csv")
    df = pd.read_csv(file_path, sep=";", decimal=",")

    # Diccionario por estación
    station_dfs = {
        est: data.drop(columns=["cod_est"]).reset_index(drop=True)
        for est, data in df.groupby("cod_est")
    }

    # Ejecutar Random Forest en todas las estaciones
    df_results = run_rf_all_stations(station_dfs)
    df_results.to_csv("rf_summary_all_stations.csv", index=False)
    print("✅ Resumen guardado en 'rf_summary_all_stations.csv'")
