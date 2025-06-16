import os
import time
import multiprocessing
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.experimental import enable_halving_search_cv  # noqa: Enables halving searches
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro
import numpy as np
from xgboost import XGBRegressor
from joblib import dump

# Número de hilos a usar (todos los cores menos uno para el sistema)
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# Carpeta de salida para los modelos
os.makedirs('modelos_xgb', exist_ok=True)

def evaluate_and_save(est, df_sta, date_col, n_splits=5):
    y = df_sta['ca_entradas']
    X = df_sta.drop(columns=['ca_entradas'])

    # Split temporal o aleatorio
    if date_col and date_col in X.columns:
        X = X.sort_values(date_col)
        y = y.loc[X.index]
        cv = TimeSeriesSplit(n_splits=n_splits)
        X = X.drop(columns=[date_col])
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Modelo XGBoost aprovechando todos los cores
    model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=N_JOBS,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=10,
        eval_metric='rmse'
    )

    # Espacio de búsqueda completo (3×3×3 = 27 combinaciones)
    param_grid = {
        'n_estimators': [100, 300, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 8]
    }

    # HalvingGridSearchCV para cubrir exhaustivamente las 27 combinaciones
    search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        factor=3,
        max_resources='auto',
        min_resources='smallest',
        aggressive_elimination=True,
        cv=cv,
        n_jobs=N_JOBS,
        scoring='neg_root_mean_squared_error',
        verbose=0
    )

    rmse_scores, r2_scores, mae_scores, residuals = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        search.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        best = search.best_estimator_

        # Guardar el modelo optimizado para esta estación
        dump(best, f'modelos_xgb/modelo_{est}.pkl')

        # Evaluación de métricas
        y_pred = best.predict(X_te)
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2_scores.append(r2_score(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))
        residuals.extend(y_te - y_pred)

    stat, p_val = shapiro(residuals)
    return {
        'cod_est': est,
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'res_shapiro_p': p_val,
        'n_feat': X.shape[1]
    }

def main():
    t0 = time.time()
    df = pd.read_csv(Path('data/dataset_metro.csv'), sep=';', decimal=',')
    date_col = 'fecha' if 'fecha' in df.columns else None

    # Crear un DataFrame por estación
    station_dfs = {
        est: data.drop(columns=['cod_est']).reset_index(drop=True)
        for est, data in df.groupby('cod_est')
    }

    resultados = []
    for est, df_sta in station_dfs.items():
        print(f'⏳ Entrenando estación {est}...')
        res = evaluate_and_save(est, df_sta, date_col)
        resultados.append(res)

    pd.DataFrame(resultados).to_csv('xgb_summary_cv_fast.csv', index=False)
    dur = (time.time() - t0) / 60
    print(f'✅ Todos los modelos entrenados y guardados en {dur:.1f} minutos.')

if __name__ == '__main__':
    main()
