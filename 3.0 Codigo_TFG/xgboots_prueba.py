import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro
import numpy as np
from xgboost import XGBRegressor
from joblib import dump
import time
import os

# Crear carpeta para guardar modelos si no existe
os.makedirs('modelos_xgb', exist_ok=True)

# === PARÁMETROS DE CONFIGURACIÓN ===
ESTACION_OBJETIVO = 1001  # <-- CAMBIA esto si quieres probar otra estación
CSV_PATH = 'data/dataset_metro.csv'
SEED = 42
N_SPLITS = 5


def entrenar_y_guardar_estacion(df_sta, cod_est, date_col=None):
    y = df_sta['ca_entradas']
    X = df_sta.drop(columns=['ca_entradas'])

    if date_col and date_col in X.columns:
        X = X.sort_values(date_col)
        y = y.loc[X.index]
        cv = TimeSeriesSplit(n_splits=N_SPLITS)
        X = X.drop(columns=[date_col])
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=10,
        eval_metric='rmse'
    )

    param_dist = {
        'n_estimators': [100, 300, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 8]
    }

    search = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        factor=3,
        max_resources='auto',
        min_resources='smallest',
        aggressive_elimination=True,
        random_state=SEED,
        cv=cv,
        n_jobs=1,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )

    rmse_scores, r2_scores, mae_scores, residuals = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        search.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        best = search.best_estimator_

        #best.save_model(f"modelos_xgb/modelo_{cod_est}.json")

        # Extraer el booster del modelo entrenado
        booster = best.get_booster()

        # Guardarlo como archivo binario (más seguro que JSON en estos casos)
        booster.save_model("modelo_1001_booster.bin")

        y_pred = best.predict(X_te)
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2_scores.append(r2_score(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))
        residuals.extend(y_te - y_pred)

    stat, p_val = shapiro(residuals)

    resumen = {
        'cod_est': cod_est,
        'R2_mean': np.mean(r2_scores),
        'RMSE_mean': np.mean(rmse_scores),
        'MAE_mean': np.mean(mae_scores),
        'res_shapiro_p': p_val,
        'n_feat': X.shape[1]
    }

    return resumen


if __name__ == '__main__':
    print(f'⏳ Iniciando entrenamiento para estación {ESTACION_OBJETIVO}...')
    t0 = time.time()

    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    date_col = 'fecha' if 'fecha' in df.columns else None

    df_sta = df[df['cod_est'] == ESTACION_OBJETIVO].drop(columns=['cod_est']).reset_index(drop=True)
    resultados = entrenar_y_guardar_estacion(df_sta, ESTACION_OBJETIVO, date_col)

    t1 = time.time()
    duracion = t1 - t0

    print(f'\n✅ Modelo entrenado y guardado para estación {ESTACION_OBJETIVO}')
    print(f'🕒 Tiempo total: {duracion:.2f} segundos')
    print('📊 Resultados:', resultados)
